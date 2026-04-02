from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml
from matplotlib import pyplot as plt

from dataset.fracture_angle_kpt_dataset import FractureAngleKeyPointDataset
from utils import inference

parser = ArgumentParser()
parser.add_argument('dataset', choices=['wrist', 'hip'])
parser.add_argument('fold', type=int)
parser.add_argument('sample', type=int)
args = parser.parse_args()


def get_line_coords(v: torch.Tensor, p0=torch.Tensor):
    t = torch.linspace(-100, 100, 100)
    line_xy = p0.unsqueeze(-2) + t.view(1, 1, -1, 1) * v.unsqueeze(-2)

    return line_xy[..., 1], line_xy[..., 0]


config_path = Path('configs/evaluation')
error = []
fold_and_index = []
match args.dataset:
    case 'wrist':
        ds = FractureAngleKeyPointDataset(args.fold, 'val', True, True)
        yaml_file = config_path / args.dataset

with open(yaml_file.with_suffix('.yaml')) as file:
    config = yaml.safe_load(file)
line_predictor = inference.KptsLinePredictor(config['heatmap_model_clearml_task_id_folds'][args.fold])

with torch.inference_mode():
    sample = ds[args.sample]
    kpts, valid = ds.extract_line_endpoints(sample['kpts'], sample['visible'])
    kpts_head = line_predictor.predict(sample['img'].unsqueeze(0))
    v_hat, p0_hat = line_predictor.convert_to_v(kpts_head, args.dataset)
    x_coord_hat, y_coord_hat = get_line_coords(v_hat, p0_hat)

    kpts[~valid] = torch.nan
    kpts = kpts.flatten(end_dim=-2)
    v, p0 = line_predictor.convert_to_v(kpts.unsqueeze(0), args.dataset)
    x_coord, y_coord = get_line_coords(v, p0)

pass
n_lines = v.shape[1]
fig, axs = plt.subplots(n_lines, 3)
for ax in axs.flatten():
    ax.imshow((sample['img'] * ds.IMG_STD + ds.IMG_MEAN).squeeze(0), 'gray')

for i in range(n_lines):
    axs[i, 0].scatter(kpts[:, 1], kpts[:, 0], marker='x')
    axs[i, 0].scatter(kpts_head[..., 1], kpts[..., 0], marker='x')
    axs[i, 1].plot(x_coord_hat[0, i], y_coord_hat[0, i], 'r:')
    axs[i, 2].plot(x_coord[0, i], y_coord[0, i], 'r:')

plt.show()
