from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml
from matplotlib import pyplot as plt
from tqdm import trange

from dataset.fracture_angle_line_dataset import FractureAngleLineDataModule
from dataset.us_hip_dataset import USHipDataModule
from dataset.autosafe_dataset import AutoSafeDataModule
from utils import eval_utils, ransac_util
from utils.inference import SegLinePredictor

parser = ArgumentParser()
parser.add_argument('dataset', choices=['wrist', 'hip', 'autosafe'])
parser.add_argument('-skeletonize', action='store_true')
args = parser.parse_args()
dataset = args.dataset

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
line_predictor = SegLinePredictor(config['segmentation_model_clearml_task_id_folds'][0], 'pca',
                                  mask_cleaning='skeletonize' if args.skeletonize else None)
line_model = line_predictor.pca_line_model

y_hat = []
v = []
with torch.inference_mode():
    for batch in dm.test_dataloader():
        x, y, valid = batch['img'], batch['mask'], batch['gt_available']
        current_y_hat = line_predictor.predict_masks(x)
        y_hat.append(current_y_hat)
        v.append(line_model.estimate(y)[0])

y_hat = torch.cat(y_hat, 0)
v = torch.cat(v, 0)
alpha, beta = eval_utils.extract_alpha_beta(v, dataset)

# ugly, quick and dirty grid search
n_loops = 100
ks = torch.arange(1, 8, step=0.25)
l1_alphas = torch.empty(n_loops, len(ks), dtype=torch.float)
l1_betas = torch.empty_like(l1_alphas)
ratio_valid = torch.empty_like(l1_alphas)
for n in trange(n_loops):
    for i, k in enumerate(ks):
        k = k.item()
        v_hat, _ = ransac_util.PointDirectionRansacLineExtractor(line_predictor.hw, 10, k).estimate(y_hat)
        alpha_hat, beta_hat = eval_utils.extract_alpha_beta(v_hat, dataset)
        l1_alphas[n, i], _, r1 = eval_utils.l1_stats_robust(alpha_hat, alpha)
        l1_betas[n, i], _, r2 = eval_utils.l1_stats_robust(beta_hat, beta)
        ratio_valid[n, i] = (r1 + r2) / 2

error = torch.stack([l1_alphas, l1_betas], dim=0).mean((0, 1))
print('L1 angle error:', error)
print('ratio valid samples', ratio_valid)
print('best k =', ks[error.argmin()].item(), 'with', error.min().item())

fig, ax1 = plt.subplots()

# Primary axis (L1 values)
ax1.plot(ks, l1_alphas.mean(0), label='L1 alpha')
ax1.plot(ks, l1_betas.mean(0), label='L1 beta')
ax1.set_ylabel('L1 values')

# Secondary axis (ratio)
ax2 = ax1.twinx()
ax2.plot(ks, ratio_valid.mean(0), 'g:', label='ratio valid samples')
ax2.set_ylabel('Valid ratio')
ax2.set_ylim(0, 1)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)
ax1.set_xlabel('K')

plt.show()
