from typing import List

import optuna
import torch
import torchmetrics
from clearml import Logger
from monai.networks import nets
from pytorch_lightning import LightningModule
from torch import nn


class HeatMapBase(LightningModule):
    def __init__(self, model: nn.Module, blob_sigma: float = 8, mse_scaling: float = 40):
        """
        Base model implementing the training routine of heatmap regression for landmark detection
        :param model: dense prediction model to train
        :param blob_sigma: control the spread of gaussian blob
        :param mse_scaling: scaling the amplitude of the gaussian blob
        """
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()
        self.sigma = blob_sigma
        self.gamma = mse_scaling

        # logger
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = self.train_loss.clone()
        self.train_error = torchmetrics.MeanAbsoluteError()
        self.val_error = self.train_error.clone()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        return optimizer

    def forward(self, x) -> torch.Tensor:
        y_hat = self.model(x)

        return y_hat

    @staticmethod
    @torch.no_grad()
    def generate_heatmaps(kpts: torch.Tensor, visible: torch.Tensor, hw: tuple[int, int], sigma: float) -> tuple[
        torch.Tensor, torch.Tensor]:
        """
        Generate heatmaps from given keypoints by providing gaussian blobs on their positions.
        :param kpts: keypoints BxKxD
        :param visible: boolean if keypoint is present BxK
        :param hw: height and width of the target resolution
        :param sigma: std of gaussian blob
        :return: heatmaps, coord
        """
        device = kpts.device
        H, W = hw
        B, K, _ = kpts.shape
        grid = torch.stack(torch.meshgrid(torch.arange(H, device=device),
                                          torch.arange(W, device=device), indexing='ij'), dim=-1)
        grid = grid.view(1, 1, H * W, 2)
        gaussian = grid - kpts.unsqueeze(2)  # (B, N, H*W, 2)
        gaussian = torch.exp(-torch.sum(gaussian.pow(2), dim=-1) / (2 * sigma ** 2))  # (B, N, H*W)

        # filter to visible keypoints and normalize
        gaussian *= visible.float().unsqueeze(-1)
        gaussian /= (gaussian.max(-1, keepdim=True).values + 1e-8)
        heatmap = gaussian.view(B, K, H, W)

        return heatmap, grid.view(1, 1, H, W, 2)

    @staticmethod
    @torch.no_grad()
    def extract_kpts_from_heatmap(heatmap: torch.Tensor, k: int = 1) -> torch.Tensor:
        """
        Extract the coordinates of the keypoints from the heatmap.
        :param heatmap: heatmap of shape (B, N, H, W)
        :param k: average over top-k coordinates
        :return:  of the keypoints of shape (B, N, 2)
        """
        B, N, H, W = heatmap.shape
        heatmap = heatmap.view(B, N, -1)
        if k == 1:
            kpts = torch.argmax(heatmap, dim=-1)
            kpts = torch.stack([kpts // W, kpts % W], dim=-1)
        else:
            top_k = torch.topk(heatmap, k, dim=-1, sorted=False)
            kpts = top_k.indices
            kpts = torch.stack([kpts // W, kpts % W], dim=-1)
            kpts = torch.sum(top_k.values.softmax(-1).unsqueeze(-1) * kpts, -2)
        return kpts

    def step(self, batch, mode):
        x, kpts, vis = batch['img'], batch['kpts'], batch['visible']
        y, _ = self.generate_heatmaps(kpts, vis, x.shape[-2:], self.sigma)

        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y * self.gamma)

        if torch.isnan(loss):
            # force one report objective value for clear-ml HPO
            Logger.current_logger().report_scalar(title="error", series="val", value=100, iteration=0)
            # let optuna drop this trial
            raise optuna.TrialPruned('NaN loss')

        # logging
        getattr(self, f'{mode}_loss').update(loss)
        if vis.any():
            kpts_hat = self.extract_kpts_from_heatmap(y_hat, k=1)
            getattr(self, f'{mode}_error').update(kpts_hat[vis], kpts[vis])

        return loss

    def log_and_reset(self, mode: str):
        loss_logger = getattr(self, f"{mode}_loss")
        avg_loss = loss_logger.compute()
        loss_logger.reset()

        error_logger = getattr(self, f'{mode}_error')
        avg_error = error_logger.compute()
        # log for pytorch_lightning to enable best model selection
        self.log('error/' + mode, avg_error, prog_bar=True, on_step=False, on_epoch=True)
        error_logger.reset()

        # log in clearml
        logger = Logger.current_logger()
        if logger is not None:
            logger.report_scalar('loss', mode, avg_loss.cpu(), self.current_epoch)
            logger.report_scalar('error', mode, avg_error.cpu(), self.current_epoch)

    def training_step(self, batch):
        return self.step(batch, 'train')

    def on_train_epoch_end(self) -> None:
        self.log_and_reset('train')

    def validation_step(self, batch):
        return self.step(batch, 'val')

    def on_validation_epoch_end(self) -> None:
        self.log_and_reset('val')


class HeatMapUNet(HeatMapBase):
    def __init__(self, in_channel: int, n_kpts: int, channels: List[int,] = [32, 64, 128, 256, 512, 1024],
                 blob_sigma: float = 8, mse_scaling: float = 40):
        model = nets.UNet(2, in_channel, n_kpts, channels, [2] * (len(channels) - 1))
        super().__init__(model, blob_sigma, mse_scaling)
        self.save_hyperparameters()
