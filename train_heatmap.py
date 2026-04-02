from shutil import rmtree

from clearml import Task
from pytorch_lightning.cli import LightningCLI

from dataset.fracture_angle_kpt_dataset import FractureAngleKeyPointDataModule # noqa
from dataset.us_hip_dataset import USHipDataModule # noqa
from dataset.autosafe_dataset import AutoSafeDataModule # noqa
from model.heatmap_models import HeatMapUNet # noqa

task = Task.init(project_name="Dislocation/HeatMap", auto_resource_monitoring=False, reuse_last_task_id=False,
                 auto_connect_frameworks=False)

# training routine
cli = LightningCLI()

# housekeeping
trainer = cli.trainer
Task.current_task().upload_artifact("best.ckpt", trainer.checkpoint_callback.best_model_path, wait_on_upload=True)
Task.current_task().upload_artifact("last.ckpt", trainer.checkpoint_callback.last_model_path, wait_on_upload=True)
Task.current_task().close()
if trainer.logger is not None:
    rmtree(trainer.logger.log_dir)

