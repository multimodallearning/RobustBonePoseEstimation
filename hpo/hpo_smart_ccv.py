from argparse import ArgumentParser
from pathlib import Path

import optuna
import torch
import yaml
from tqdm import tqdm

from dataset.fracture_angle_line_dataset import FractureAngleLineDataModule
from dataset.us_hip_dataset import USHipDataModule
from dataset.autosafe_dataset import AutoSafeDataModule
from utils import eval_utils, false_positiv_reduction
from utils.inference import SegLinePredictor

parser = ArgumentParser()
parser.add_argument('dataset', choices=['wrist', 'hip', 'autosafe'])
dataset = parser.parse_args().dataset

config_path = Path('configs/evaluation')
match dataset:
    case 'wrist':
        dm = FractureAngleLineDataModule(0, 32)
    case 'hip':
        dm = USHipDataModule(0, 32)
    case 'autosafe':
        dm = AutoSafeDataModule(0, 32)

dm.setup('test')
yaml_file = config_path / dataset
with open(yaml_file.with_suffix('.yaml')) as file:
    config = yaml.safe_load(file)
line_predictor = SegLinePredictor(config['segmentation_model_clearml_task_id_folds'][0], 'pca')
line_model = line_predictor.pca_line_model

y_hat = []
v = []
with torch.inference_mode():
    for batch in tqdm(dm.test_dataloader()):
        x, y, valid = batch['img'], batch['mask'], batch['gt_available']
        current_y_hat = line_predictor.predict_masks(x)
        y_hat.append(current_y_hat)
        v.append(line_model.estimate(y)[0])

y_hat = torch.cat(y_hat, 0)
v = torch.cat(v, 0)
alpha, beta = eval_utils.extract_alpha_beta(v, dataset)


def objective(trial: optuna.Trial):
    thr_min_abs_area = trial.suggest_int('thr_min_abs_area', 1, 1001, step=10)
    thr_min_rel_area = trial.suggest_float('thr_min_rel_area', 0.01, 1., step=0.01)
    cleaned_masks = false_positiv_reduction.smart_cca(y_hat, thr_min_abs_area, thr_min_rel_area)
    v_hat = line_model.estimate(cleaned_masks)[0]
    alpha_hat, beta_hat = eval_utils.extract_alpha_beta(v_hat, dataset)
    l1_alpha, _, r_alpha = eval_utils.l1_stats_robust(alpha_hat, alpha)
    l1_betas, _, r_betas = eval_utils.l1_stats_robust(beta_hat, beta)
    l1 = (l1_alpha + l1_betas) / 2
    r = (r_alpha + r_betas) / 2

    return l1, r


study = optuna.create_study(directions=['minimize', 'maximize'],
                            storage=f"sqlite:///hpo/{dataset}.sqlite3", study_name=f"{dataset}_smart_ccv")
study.optimize(objective, n_trials=300)
