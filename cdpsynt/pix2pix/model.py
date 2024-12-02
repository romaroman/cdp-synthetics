from collections import OrderedDict
from typing import Any

from omegaconf import DictConfig
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torchmetrics.functional import accuracy
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image.ssim import (
    structural_similarity_index_measure as ssim,
)
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from .unet import UNetModel
from .discriminator import Discriminator
from .generator import Generator

__all__ = ["Pix2PixWF"]


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.contiguous()
        targets = targets.contiguous()

        intersection = (inputs * targets).sum(dim=2).sum(dim=2)
        dice = (2.0 * intersection + smooth) / (
            inputs.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2) + smooth
        )

        return 1 - dice.mean()


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, inputs, targets):
        gradient_inputs_x = torch.abs(inputs[:, :, :-1, :] - inputs[:, :, 1:, :])
        gradient_inputs_y = torch.abs(inputs[:, :, :, :-1] - inputs[:, :, :, 1:])

        gradient_targets_x = torch.abs(targets[:, :, :-1, :] - targets[:, :, 1:, :])
        gradient_targets_y = torch.abs(targets[:, :, :, :-1] - targets[:, :, :, 1:])

        gradient_inputs_x = gradient_inputs_x[:, :, :-1, :]
        gradient_targets_x = gradient_targets_x[:, :, :-1, :]

        gradient_inputs_y = gradient_inputs_y[:, :, :, :-1]
        gradient_targets_y = gradient_targets_y[:, :, :, :-1]

        gradient_diff_x = torch.abs(gradient_inputs_x - gradient_targets_x)
        gradient_diff_y = torch.abs(gradient_inputs_y - gradient_targets_y)

        return torch.mean(gradient_diff_x) + torch.mean(gradient_diff_y)


class CombinedLoss(nn.Module):
    def __init__(
        self,
        loss_weights: DictConfig | None = None,
        
    ):
        super(CombinedLoss, self).__init__()
        loss_weights = loss_weights or DictConfig({})
        self.weight_dice = loss_weights.get('dice', 1)
        self.weight_bce = loss_weights.get('BCE', 1)

        self.weight_gradient = loss_weights.get('gradient', 10)
        self.weight_ssim = loss_weights.get('ssim', 50)
        self.weight_l1 = loss_weights.get('l1', 10)
        self.weight_l2 = loss_weights.get('l2', 10)

        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.gradient_loss = GradientLoss()
        self.ssim_loss = SSIM(sigma=2.5, data_range=(0, 1))

    def forward(self, prediction_image, target_image, prediction_label, target_label):
        dice = self.dice_loss(prediction_label, target_label)
        bce = self.bce_loss(prediction_label, target_label)
        
        gradient = self.gradient_loss(prediction_image, target_image)
        ssim = 1 - self.ssim_loss(prediction_image, target_image)
        l2 = F.mse_loss(prediction_image, target_image)
        l1 = F.l1_loss(prediction_image, target_image)

        combined = (
            self.weight_dice * dice
            + self.weight_bce * bce
            + self.weight_gradient * gradient
            + self.weight_ssim * ssim
            + self.weight_l1 * l1
            + self.weight_l2 * l2
        )

        return combined, dice, bce, gradient, ssim, l2, l1


def batch_pearson_correlation(batch1, batch2):
    assert batch1.shape == batch2.shape, "Input batches must have the same shape"
    B, C, H, W = batch1.shape

    batch1_flattened = batch1.view(B, C, -1)
    batch2_flattened = batch2.view(B, C, -1)

    mean1 = torch.mean(batch1_flattened, dim=-1, keepdim=True)
    mean2 = torch.mean(batch2_flattened, dim=-1, keepdim=True)
    batch1_centered = batch1_flattened - mean1
    batch2_centered = batch2_flattened - mean2

    covariance = torch.sum(batch1_centered * batch2_centered, dim=-1)

    std1 = torch.sqrt(torch.sum(batch1_centered**2, dim=-1))
    std2 = torch.sqrt(torch.sum(batch2_centered**2, dim=-1))

    correlation = covariance / (std1 * std2)

    return correlation


def batch_mse(prediction, target):
    return ((prediction - target) ** 2).mean(dim=(1, 2, 3))


class Pix2PixWF(pl.LightningModule):
    def __init__(
        self,
        generator_backbone: str = "my",
        discriminator_dropout_p: float = 0.4,
        generator_dropout_p: float = 0.4,
        generator_lr: float = 1e-4,
        discriminator_lr: float = 1e-5,
        weight_decay: float = 1e-5,
        lr_scheduler_T_0: float = 1e3,
        lr_scheduler_T_mult: int = 2,
        generator_channels_in: int = 1,
        generator_channels_out: int = 1,
        discriminator_channels_in: int = 2,
        use_auc: bool = False,
        snp_prob: float = 0.0,
        loss_weights: DictConfig | None = None,
    ):
        super(Pix2PixWF, self).__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.weight_decay = weight_decay
        self.lr_scheduler_T_0 = lr_scheduler_T_0
        self.lr_scheduler_T_mult = lr_scheduler_T_mult

        self.use_auc: bool = use_auc
        self.snp_prob: float = snp_prob

        # Models
        if generator_backbone == "my":
            self.generator = UNetModel(generator_channels_in, generator_channels_out)
        elif generator_backbone == "smp":
            self.generator = Generator(
                generator_channels_in, generator_channels_out, generator_dropout_p
            )

        self.discriminator = Discriminator(
            discriminator_channels_in, discriminator_dropout_p
        )

        self.generator_loss = CombinedLoss(loss_weights)

        self.test_df = None

        self.test_sims = []
        self.train_sims = []
        self.val_sims = []

    def forward(self, x):
        return self.generator(x)

    def discriminator_loss(self, prediction_label, target_label):
        bce_loss = F.binary_cross_entropy(prediction_label, target_label)
        return bce_loss

    def configure_optimizers(self):
        # Optimizers
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.generator_lr,
            weight_decay=self.weight_decay,
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.discriminator_lr,
            weight_decay=self.weight_decay,
        )
        # Learning Scheduler
        genertator_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            generator_optimizer,
            T_0=int(self.lr_scheduler_T_0),
            T_mult=int(self.lr_scheduler_T_mult),
        )
        discriminator_lr_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                discriminator_optimizer,
                T_0=int(self.lr_scheduler_T_0),
                T_mult=int(self.lr_scheduler_T_mult),
            )
        )
        return [generator_optimizer, discriminator_optimizer], [
            genertator_lr_scheduler,
            discriminator_lr_scheduler,
        ]

    def training_step(self, batch, batch_idx):
        # Optimizers
        generator_optimizer, discriminator_optimizer = self.optimizers()
        generator_lr_scheduler, discriminator_lr_scheduler = self.lr_schedulers()

        orig_template = batch["t"]

        if self.snp_prob:
            bitflip_mask = torch.rand_like(orig_template) < self.snp_prob
            orig_template[bitflip_mask] = 1 - orig_template[bitflip_mask]

        orig_physical_real = batch["x"]

        orig_template_gen, orig_template_dis = torch.split(
            orig_template, len(orig_template) // 2
        )
        orig_physical_real_gen, orig_physical_real_dis = torch.split(
            orig_physical_real, len(orig_physical_real) // 2
        )

        # Generator Feed-Forward
        orig_physical_synth = torch.clip(self.forward(orig_template_gen), 0, 1)

        ######################################
        #  Discriminator Loss and Optimizer  #
        ######################################
        # Discriminator Feed-Forward
        discriminator_prediction_real = self.discriminator(
            torch.cat((orig_template_gen, orig_physical_real_gen), dim=1)
        )
        discriminator_prediction_synth = self.discriminator(
            torch.cat((orig_template_gen, orig_physical_synth), dim=1)
        )

        # Discriminator Loss
        discriminator_label_real = self.discriminator_loss(
            discriminator_prediction_real,
            torch.ones_like(discriminator_prediction_real),
        )
        discriminator_label_synth = self.discriminator_loss(
            discriminator_prediction_synth,
            torch.zeros_like(discriminator_prediction_synth),
        )
        discriminator_loss = discriminator_label_real + discriminator_label_synth

        # Discriminator Optimizer
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()
        discriminator_lr_scheduler.step()

        ##################################
        #  Generator Loss and Optimizer  #
        ##################################
        orig_physical_synth_dis = torch.clip(self.forward(orig_template_dis), 0, 1)
        discriminator_prediction_synth = self.discriminator(
            torch.cat((orig_template_dis, orig_physical_synth_dis), dim=1)
        )

        # Generator loss
        generator_comb_loss, dice_loss, bce_loss, gradient_loss, ssim_loss, l2_loss, l1_loss = self.generator_loss(
            orig_physical_synth_dis,
            orig_physical_real_dis,
            discriminator_prediction_synth,
            torch.ones_like(discriminator_prediction_synth),
        )

        # Generator Optimizer
        generator_optimizer.zero_grad()
        generator_comb_loss.backward()
        generator_optimizer.step()
        generator_lr_scheduler.step()

        # Progressbar and Logging
        loss = OrderedDict(
            {
                "train_g_dice_loss": dice_loss,
                "train_g_bce_loss": bce_loss,
                "train_g_gradient_loss": gradient_loss,
                "train_g_ssim_loss": ssim_loss,
                "train_g_l2_loss": l2_loss,
                "train_g_l1_loss": l1_loss,
                "train_g_comb_loss": generator_comb_loss,
                "train_d_loss": discriminator_loss,
                "train_d_loss_real": discriminator_label_real,
                "train_d_loss_fake": discriminator_label_synth,
                "train_g_lr": generator_lr_scheduler.get_last_lr()[0],
                "train_d_lr": discriminator_lr_scheduler.get_last_lr()[0],
            }
        )

        if self.use_auc:
            self.train_sims.extend(
                self.assess_similarity(
                    batch,
                    torch.cat((orig_physical_synth, orig_physical_synth_dis), dim=0),
                )
            )

        self.log_dict(loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        orig_template = batch["t"]
        orig_physical_real = batch["x"]

        # Generator Feed-Forward
        orig_physical_synth = self.forward(orig_template)
        orig_physical_synth = torch.clip(orig_physical_synth, 0, 1)

        # Generator Metrics
        generator_psnr = psnr(orig_physical_synth, orig_physical_real)
        generator_ssim = ssim(orig_physical_synth, orig_physical_real)

        discriminator_prediction_synth = self.discriminator(
            torch.cat((orig_template, orig_physical_synth), dim=1)
        )
        generator_accuracy = accuracy(
            discriminator_prediction_synth,
            torch.ones_like(discriminator_prediction_synth, dtype=torch.int32),
            "binary",
        )

        # Discriminator Feed-Forward
        discriminator_prediction_real = self.discriminator(
            torch.cat((orig_template, orig_physical_real), dim=1)
        )
        discriminator_prediction_synth = self.discriminator(
            torch.cat((orig_template, orig_physical_synth), dim=1)
        )

        real_accuracy = accuracy(
            discriminator_prediction_real,
            torch.ones_like(discriminator_prediction_real, dtype=torch.int32),
            "binary",
        )

        synth_accuracy = accuracy(
            discriminator_prediction_synth,
            torch.zeros_like(discriminator_prediction_synth, dtype=torch.int32),
            "binary",
        )

        # Discriminator Metrics
        discriminator_accuracy = real_accuracy / 2 + synth_accuracy / 2
        if self.use_auc:
            self.val_sims.extend(self.assess_similarity(batch, orig_physical_synth))

        # Progressbar and Logging
        metrics = OrderedDict(
            {
                "val_g_psnr": generator_psnr,
                "val_g_ssim": generator_ssim,
                "val_g_accuracy": generator_accuracy,
                "val_d_accuracy": discriminator_accuracy,
                "val_d_accuracy_fake": synth_accuracy,
                "val_d_accuracy_real": real_accuracy,
            }
        )

        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx) -> None:
        self.test_sims.extend(self.assess_similarity(batch, self.forward(batch["t"])))

    def assess_similarity(self, batch, template_synthetic) -> list[dict[str, Any]]:
        orig_physical, template_digital = batch["x"], batch["t"]

        batch_ssim = SSIM(reduction="none", sigma=2.5, data_range=(0, 1)).to(
            orig_physical.device
        )
        batch_psnr = PSNR(reduction="none", dim=(2, 3), data_range=(0, 1)).to(
            orig_physical.device
        )

        vers = {
            "orig": orig_physical,
        }
        if "f" in batch:
            vers["fake"] = batch["f"]

        refs = {
            "digital": template_digital,
            "synthetic": template_synthetic,
        }
        if "xref" in batch:
            refs["physical"] = batch["xref"]

        metrics = {
            "ssim": batch_ssim,
            "pcorr": batch_pearson_correlation,
            "mse": batch_mse,
            "psnr": batch_psnr,
        }

        result = self.calc_similarity(vers, refs, metrics)
        result = result | {
            "uuid": batch["uuid"],
            "shot": batch["shot"],
            "block": batch["block"],
        }

        return [
            {
                key: value[i].item() if isinstance(value[i], torch.Tensor) else value[i]
                for key, value in result.items()
            }
            for i in range(batch["x"].shape[0])
        ]

    def calc_similarity(
        self,
        vers: dict[str, torch.Tensor],
        refs: dict[str, torch.Tensor],
        metrics: dict,
    ) -> dict[str, torch.Tensor]:
        result = {}
        for metric, metric_func in metrics.items():
            for ref, ref_tensor in refs.items():
                for ver, ver_tensor in vers.items():
                    result["_".join([metric, ver, ref])] = metric_func(
                        ver_tensor, ref_tensor
                    )
        return result

    def calc_aucs(
        self, sims: list[dict[str, Any]], prefix: str
    ) -> OrderedDict[str, float]:
        df = pd.DataFrame(sims)

        dfm = df.melt(
            id_vars=["uuid", "shot", "block"],
            var_name="measurement",
            value_name="score",
        )
        dfm["metric"], dfm["origin"], dfm["reference"] = zip(
            *dfm["measurement"].apply(lambda x: x.split("_"))
        )

        dff = dfm.pivot_table(
            index=["uuid", "shot", "block", "origin", "reference"],
            columns="metric",
            values="score",
        ).reset_index()

        dff.columns.name = None
        dff = dff.groupby(["uuid", "shot", "origin", "reference"]).mean().reset_index()
        dff.drop(columns=["block"], inplace=True)

        aucs = OrderedDict()
        for ref, df_ref in dff.groupby("reference"):
            for metric in ["ssim", "pcorr", "mse", "psnr"]:
                if not metric in df_ref.columns:
                    continue
                auc_val = roc_auc_score(
                    df_ref["origin"] == "orig", df_ref[metric].values
                )
                aucs["_".join([prefix, "auc", metric, ref])] = (
                    auc_val if auc_val > 0.5 else 1 - auc_val
                )

        return aucs

    def on_test_epoch_end(self):
        self.test_df = pd.DataFrame(self.test_sims)
        self.test_sims = []

    def on_validation_epoch_end(self) -> None:
        if self.use_auc:
            self.log_dict(self.calc_aucs(self.val_sims, "val"), prog_bar=True)
            self.val_sims = []

    def on_train_epoch_end(self) -> None:
        if self.use_auc:
            self.log_dict(self.calc_aucs(self.train_sims, "train"), prog_bar=True)
            self.train_sims = []
