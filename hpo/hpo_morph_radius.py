from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from dataset.fracture_angle_line_dataset import FractureAngleLineDataModule
from dataset.us_hip_dataset import USHipDataModule
from dataset.autosafe_dataset import AutoSafeDataModule
from utils import eval_utils, false_positiv_reduction
from utils.inference import SegLinePredictor

parser = ArgumentParser()
parser.add_argument('dataset', choices=['wrist', 'hip', 'autosafe'])
parser.add_argument('preprocessor', choices=['opening', 'cv_pipeline'])
parser = parser.parse_args()
dataset = parser.dataset

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

# ugly, quick and dirty grid search
radii = torch.arange(1, 8)
l1_alphas = torch.empty_like(radii, dtype=torch.float)
l1_betas = torch.empty_like(l1_alphas)
ratio_valid = torch.empty_like(l1_alphas)
for i, radius in enumerate(tqdm(radii)):
    radius = radius.item()
    if parser.preprocessor == 'opening':
        cleaned_masks = false_positiv_reduction.opening(y_hat, radius)
    else:
        cleaned_masks = false_positiv_reduction.opening_closing_cca(y_hat, radius)
    v_hat = line_model.estimate(cleaned_masks)[0]
    alpha_hat, beta_hat = eval_utils.extract_alpha_beta(v_hat, parser.dataset)
    l1_alphas[i], _,  r1 = eval_utils.l1_stats_robust(alpha_hat, alpha)
    l1_betas[i], _, r2 = eval_utils.l1_stats_robust(beta_hat, beta)
    ratio_valid[i] = (r1 + r2) / 2

print('L1 angle error:', torch.stack([l1_alphas, l1_betas], dim=0).mean(0))
print('ratio valid samples', ratio_valid)

fig, ax1 = plt.subplots()

# Primary axis (L1 values)
ax1.plot(radii, l1_alphas, label='L1 alpha')
ax1.plot(radii, l1_betas, label='L1 beta')
ax1.set_ylabel('L1 values')

# Secondary axis (ratio)
ax2 = ax1.twinx()
ax2.plot(radii, ratio_valid, 'g:', label='ratio valid samples')
ax2.set_ylabel('Valid ratio')
ax2.set_ylim(0, 1)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)
ax1.set_xlabel('Radius')

plt.show()


