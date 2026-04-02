from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt

from dataset.fracture_angle_line_dataset import FractureAngleLineDataset
from dataset.us_hip_dataset import USHipDataset
from dataset.autosafe_dataset import AutoSafeDataset
from utils import inference
import numpy as np

parser = ArgumentParser()
parser.add_argument('dataset', choices=['wrist', 'hip', 'autosafe'])
parser.add_argument('mask_cleaning', choices=['cca', 'opening', 'cv_pipeline', 'skeletonize', 'none'])
parser.add_argument('line_extractor', choices=['ransac', 'hough', 'pca'])
parser.add_argument('fold', type=int)
parser.add_argument('sample', type=int)
args = parser.parse_args()

config_path = Path('configs/evaluation')
match args.dataset:
    case 'wrist':
        ds = FractureAngleLineDataset(args.fold, 'val', True, True)
        figsize = (15, 5)
    case 'hip':
        ds = USHipDataset(args.fold, 'val', z_std_img=True)
        figsize = (15, 3)
    case 'autosafe':
        ds = AutoSafeDataset(args.fold, 'val', True)
        figsize = (15, 3)

with open((config_path / args.dataset).with_suffix('.yaml')) as file:
    config = yaml.safe_load(file)
line_predictor = inference.SegLinePredictor(config['segmentation_model_clearml_task_id_folds'][args.fold],
                                            args.line_extractor, config['ransac_k'], args.mask_cleaning,
                                            config['opening_radius'] if args.mask_cleaning else config[
                                                'cv_pipeline_radius'],
                                            config['connected_component_analyse']['min_abs_area'],
                                            config['connected_component_analyse']['min_rel_to_biggest_area'])
line_model = line_predictor.pca_line_model

with torch.inference_mode():
    sample = ds[args.sample]
    x, y, valid = sample['img'], sample['mask'], sample['gt_available']
    y_hat, y_hat_cleaned, (x_coord_hat, y_coord_hat) = line_predictor.predict_for_plotting(x.unsqueeze(0))
    x_coord, y_coord = line_model.get_line_coords(y.unsqueeze(0))

fig, axs = plt.subplots(1, 5, figsize=figsize)
for ax in axs.flatten():
    ax.imshow((x * ds.IMG_STD + ds.IMG_MEAN).squeeze(0), 'gray')
    ax.axis('off')
    ax.set_autoscale_on(False)


def convert_to_colored_image(mask: torch.Tensor, color: np.ndarray):
    colored = np.zeros((*mask.shape, 4))
    colored[..., :3] = color[:3]  # RGB color
    colored[..., 3] = mask.float().numpy()  # alpha = mask intensity

    return colored


n_lines = y.shape[0]
colors = plt.cm.tab10(np.linspace(0, 1, n_lines))
for i in range(n_lines):
    axs[0].imshow(convert_to_colored_image(y_hat[0, i], colors[i]))
    axs[1].imshow(convert_to_colored_image(y_hat_cleaned[0, i], colors[i]))
    axs[2].imshow(convert_to_colored_image(y_hat_cleaned[0, i], colors[i]))
    axs[2].plot(x_coord_hat[0, i], y_coord_hat[0, i], 'r')
    axs[3].imshow(convert_to_colored_image(y_hat_cleaned[0, i], colors[i]))
    axs[3].plot(x_coord[0, i], y_coord[0, i], 'b:')
    axs[3].plot(x_coord_hat[0, i], y_coord_hat[0, i], 'r')
fig.savefig(
    f'/home/ron/Documents/Konferenzen/MIAU/{args.dataset}_{args.mask_cleaning}_{args.line_extractor}_{args.fold}_{args.sample}.png',
    dpi=600)
plt.show()
