
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


def select_gpu_or_cpu(cfg: DictConfig) -> int | None:
    """
    Returns GPU ID if requested or available, otherwise None for CPU.
    """
    if cfg.train.gpu != "auto":
        # If explicitly set to 'cpu', use CPU
        if str(cfg.train.gpu).lower() == "cpu":
            return None
        # If it's an integer, we assume GPU with that ID is accessible
        try:
            return int(cfg.train.gpu)
        except ValueError:
            raise ValueError(f"Unexpected gpu config value: {cfg.train.gpu}")

    # 'auto' mode: pick the GPU with the most free memory, or fallback to CPU if none
    gpus = GPUtil.getGPUs()
    return max(gpus, key=lambda x: x.memoryFree).id if gpus else None


def get_session(cfg: DictConfig) -> str:
    session = "_".join(map(str, [cfg.dataset.device, cfg.dataset.printer]))
    if "prefix_session" in cfg and cfg.prefix_session:
        session = f"{cfg.prefix_session}_{session}"
    return session


def get_dir_out(cfg: DictConfig) -> Path:
    return Path(cfg.io.dir_root) / "out" / cfg.prefix / get_session(cfg)


def init_logger(cfg: DictConfig) -> WandbLogger:
    kwargs_wandb = {"reinit": False}
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
    """
    Creates an instance of Pix2PixWF with the parameters from config.
    """
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


def init_callbacks(cfg: DictConfig, test_dataloader: DataLoader) -> list:
    """
    Returns the list of callbacks for the PyTorch Lightning Trainer.
    """
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=get_dir_out(cfg) / "checkpoints",
        filename="{epoch:02d}_{" + cfg.checkpoint.metric + ":.3f}",
        monitor=cfg.checkpoint.metric,
        mode=cfg.checkpoint.mode,
        every_n_epochs=cfg.checkpoint.every,
        verbose=True,
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "_"

    return [
        checkpoint_callback,
        EpochInference(test_dataloader, dir_dst=get_dir_out(cfg) / "inferences", n_imgs=8),
    ]


def run_train_model(cfg: DictConfig, dataset: DatasetNN) -> None:
    """
    Train the Pix2PixWF model with the given dataset and config.
    """
    callbacks = init_callbacks(cfg, dataset.test_dataloader())
    model = init_model(cfg)

    # If GPU is None => CPU
    accelerator = "gpu" if GPU is not None else "cpu"
    devices = [GPU] if GPU is not None else 1

    trainer = pl.Trainer(
        logger=init_logger(cfg),
        log_every_n_steps=cfg.io.wandb.log_every,
        callbacks=callbacks,
        max_epochs=cfg.train.max_epochs,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg.train.validation_every,
        devices=devices,
        accelerator=accelerator,
    )

    trainer.fit(
        model,
        train_dataloaders=dataset.train_dataloader(),
        val_dataloaders=dataset.val_dataloader(),
    )


def run_test_model(cfg: DictConfig, dataset: DatasetNN) -> None:
    """
    Test the Pix2PixWF model with the given dataset and config.
    """
    path_model = list((get_dir_out(cfg) / "checkpoints").glob("*.ckpt"))[0]
    map_loc = f"cuda:{GPU}" if GPU is not None else "cpu"

    model = Pix2PixWF.load_from_checkpoint(path_model, map_location=map_loc, loss_weights=cfg.model.generator.loss)

    accelerator = "gpu" if GPU is not None else "cpu"
    devices = [GPU] if GPU is not None else 1
    trainer = pl.Trainer(accelerator=accelerator, devices=devices)
    trainer.test(model, dataset.test_dataloader())

    path_df = get_dir_out(cfg) / "results" / "test.csv"
    path_df.parent.mkdir(parents=True, exist_ok=True)
    model.test_df.to_csv(path_df)


@hydra.main(
    version_base=None,
    config_path=f"{DIR_PROJECT.resolve()}/config",
    config_name="general",
)
def main(cfg: DictConfig) -> None:
    dataset = load_dataset(cfg)

    global GPU
    GPU = select_gpu_or_cpu(cfg)

    if cfg.mode == "train":
        run_train_model(cfg, dataset)
    elif cfg.mode == "test":
        run_test_model(cfg, dataset)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()