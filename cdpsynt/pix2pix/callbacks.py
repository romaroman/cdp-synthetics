from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision

__all__ = ["EpochInference"]


class EpochInference(pl.Callback):
    def __init__(self, dataloader, dir_dst: Path, n_imgs: int = 10, *args, **kwargs):
        super(EpochInference, self).__init__(*args, **kwargs)
        self.dataloader = dataloader
        self.dir_dst: Path = dir_dst
        self.dir_dst.mkdir(parents=True, exist_ok=True)
        self.n_imgs: int = n_imgs

    def on_train_epoch_end(self, trainer, pl_module):
        super(EpochInference, self).on_train_epoch_end(trainer, pl_module)
        batch = next(iter(self.dataloader))
        image = batch["t"][:self.n_imgs]
        target = batch["x"][:self.n_imgs]

        image = image.cuda()
        target = target.cuda()
        with torch.no_grad():
            reconstruction_init = pl_module.forward(image)
            reconstruction_init = torch.clip(reconstruction_init, 0, 1)
            # reconstruction_mean = torch.stack(
            #     [pl_module.forward(image) for _ in range(30)]
            # )
            # reconstruction_mean = torch.clip(reconstruction_mean, 0, 1)
            # reconstruction_mean = torch.mean(reconstruction_mean, dim=0)

        if image.shape[1] != target.shape[1]:
            image = torch.stack([image for _ in range(target.shape[1])], dim=1)
            image = torch.squeeze(image)
        
        grid_image = torchvision.utils.make_grid(
            torch.cat([image, target, reconstruction_init], dim=0), # , reconstruction_mean
            nrow=self.n_imgs,
        )
        torchvision.utils.save_image(
            grid_image, fp=self.dir_dst / f"epoch-{trainer.current_epoch:04}.png"
        )
