from typing import List

import torch
import torchmetrics
from clearml import Logger
from monai.networks import nets
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
from torchmetrics import classification


class LineSegBase(LightningModule):
    def __init__(self, model: nn.Module, pos_weight: int = 50):
        """
        Base model implementing the training routine for a multilabel segmentation model
        """
        super().__init__()
        self.model: nn.Module = model
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).view(1, 1, 1), reduction='none')

        # logger
        self.train_loss = torchmetrics.MeanMetric(nan_strategy='error')
        self.val_loss = self.train_loss.clone()
        # binary = micro-averaged multilabel metric
        self.train_error = torchmetrics.MetricCollection({
            "F1": classification.BinaryF1Score(),
            "Precision": classification.BinaryPrecision(),
            "Recall": classification.BinaryRecall()
        }, postfix='/train')
        self.val_error = self.train_error.clone(postfix='/val')

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.model.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.model(x)
        return y_hat

    def step(self, batch, mode: str):
        if isinstance(batch, list):
            x, y = batch
            y = y.unsqueeze(1)
            gt_available = torch.ones(len(x), 1, dtype=torch.bool, device=self.device)
        elif isinstance(batch, dict):
            x, y, gt_available = batch['img'], batch['mask'], batch['gt_available']
        else:
            raise RuntimeError(f'Unexpected instance of batch: {type(batch)}')

        y_masked = y[gt_available].float()
        y_hat = self.forward(x)
        y_hat_masked = y_hat[gt_available]
        loss = self.criterion(y_hat_masked, y_masked).mean()

        # logging
        getattr(self, f'{mode}_loss').update(loss)
        getattr(self, f'{mode}_error').update(y_hat, y)

        return loss

    def log_and_reset(self, mode: str):
        clearml_logger: Logger = Logger.current_logger()
        if clearml_logger is None: return

        loss_logger = getattr(self, f"{mode}_loss")
        avg_loss = loss_logger.compute()
        loss_logger.reset()
        clearml_logger.report_scalar('loss', mode, avg_loss.cpu(), self.trainer.global_step)

        metric_logger = getattr(self, f"{mode}_error")
        metrics = metric_logger.compute()
        metric_logger.reset()
        for name, value in metrics.items():
            name = name.split('/')[0]
            clearml_logger.report_scalar(name, mode, value.cpu(), self.trainer.global_step)
            self.log('/'.join([name, mode]), value, on_step=False, on_epoch=True)

    def training_step(self, batch):  # no epochs for training data
        loss = self.step(batch, 'train')
        if (self.global_step + 1) % self.trainer.log_every_n_steps == 0: self.log_and_reset('train')

        return loss

    def validation_step(self, batch):
        return self.step(batch, 'val')

    def on_validation_epoch_end(self):
        self.log_and_reset('val')


class LineUNet(LineSegBase):
    def __init__(self, pos_weight: int = 1, channels: List[int,] = [32, 64, 128, 256, 512], in_channel: int = 1,
                 out_channel: int = 4):
        model = nets.UNet(2, in_channel, out_channel, channels, [2] * (len(channels) - 1))
        super().__init__(model, pos_weight)
        self.save_hyperparameters()
