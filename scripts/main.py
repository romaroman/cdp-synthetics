from pathlib import Path

import GPUtil
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from cdpsynt.common import DIR_PROJECT
from cdpsynt.data import DatasetNN
from cdpsynt.dataset import load_dataset
from cdpsynt.pix2pix import EpochInference, Pix2PixWF

torch.set_float32_matmul_precision("high")

GPU = None


def select_gpu():
    gpus = GPUtil.getGPUs()
    if not gpus:
        raise RuntimeError("No GPUs available")

    gpu = max(gpus, key=lambda x: x.memoryFree)
    return gpu.id


def get_session(cfg: DictConfig) -> str:
    session = "_".join(map(str, [cfg.dataset.device, cfg.dataset.printer]))
    if "prefix_session" in cfg and cfg.prefix_session:
        session = f"{cfg.prefix_session}_{session}"
    return session


def get_dir_out(cfg: DictConfig) -> Path:
    return Path(cfg.io.dir_root) / "out" / cfg.prefix / get_session(cfg)


def init_logger(cfg: DictConfig):
    kwargs_wandb = {
        "reinit": False,
    }
    if not cfg.io.wandb.enabled:
        kwargs_wandb["mode"] = "disabled"

    logger_wandb = WandbLogger(
        project=f"{cfg.io.wandb.project}.{cfg.prefix}",
        name=get_session(cfg),
        **kwargs_wandb,
    )
    logger_wandb.log_hyperparams(dict(cfg.train))
    logger_wandb.log_hyperparams(dict(cfg.model))
    logger_wandb.log_hyperparams(dict(cfg.dataset))
    return logger_wandb


def init_model(cfg: DictConfig) -> Pix2PixWF:
    return Pix2PixWF(
        generator_backbone=cfg.model.generator.backbone,
        weight_decay=cfg.model.decay,
        lr_scheduler_T_0=int(len(load_dataset(cfg).train_dataloader()) * 1.5),
        lr_scheduler_T_mult=2,
        generator_channels_in=cfg.model.generator.channels_in,
        generator_channels_out=cfg.model.generator.channels_out,
        generator_dropout_p=cfg.model.dropout,
        generator_lr=cfg.model.generator.lr,
        discriminator_channels_in=cfg.model.discriminator.channels_in,
        discriminator_dropout_p=cfg.model.dropout,
        discriminator_lr=cfg.model.discriminator.lr,
        snp_prob=cfg.model.generator.snp,
        use_auc=cfg.dataset.fake,
        loss_weights=cfg.model.generator.loss,
    )


def init_callbacks(cfg: DictConfig, test_dataloader: DataLoader) -> pl.Trainer:
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=get_dir_out(cfg) / "checkpoints",
        filename="{epoch:02d}_{" + cfg.checkpoint.metric + ":.3f}",
        monitor=cfg.checkpoint.metric,
        mode=cfg.checkpoint.mode,
        every_n_epochs=cfg.checkpoint.every,
        verbose=True,
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "_"
    callbacks = [
        checkpoint_callback,
        EpochInference(
            test_dataloader, dir_dst=get_dir_out(cfg) / "inferences", n_imgs=8
        ),
    ]
    return callbacks


def train(cfg: DictConfig, dataset: DatasetNN) -> None:
    callbacks = init_callbacks(cfg, dataset.test_dataloader())
    model = init_model(cfg)

    trainer = pl.Trainer(
        logger=init_logger(cfg),
        log_every_n_steps=cfg.io.wandb.log_every,
        callbacks=callbacks,
        max_epochs=cfg.train.max_epochs,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg.train.validation_every,
        devices=[GPU],
        accelerator="gpu",
    )

    trainer.fit(
        model,
        train_dataloaders=dataset.train_dataloader(),
        val_dataloaders=dataset.val_dataloader(),
    )


def test(cfg: DictConfig, dataset: DatasetNN) -> None:
    path_model = list((get_dir_out(cfg) / "checkpoints").glob("*.ckpt"))[0]
    model = Pix2PixWF.load_from_checkpoint(
        path_model, map_location=f"cuda:{GPU}", loss_weights=cfg.model.generator.loss
    )

    trainer = pl.Trainer(devices=[GPU], accelerator="gpu")
    trainer.test(model, dataset.test_dataloader())

    path_df = get_dir_out(cfg) / "results" / "test.csv"
    path_df.parent.mkdir(parents=True, exist_ok=True)
    model.test_df.to_csv(path_df)


@hydra.main(
    version_base=None,
    config_path=f"{DIR_PROJECT.rescolve()}/config",
    config_name="train",
)
def main(cfg: DictConfig):
    dataset = load_dataset(cfg)
    global GPU
    GPU = cfg.train.gpu if cfg.train.gpu != "auto" else select_gpu()

    if cfg.mode == "train":
        train(cfg, dataset)
    elif cfg.mode == "test":
        test(cfg, dataset)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
