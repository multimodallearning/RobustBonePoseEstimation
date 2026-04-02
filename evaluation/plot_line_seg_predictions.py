from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from tqdm import trange

from dataset.fracture_angle_line_dataset import FractureAngleLineDataset
from dataset.us_hip_dataset import USHipDataset
from utils import inference

parser = ArgumentParser()
parser.add_argument('dataset', choices=['wrist', 'hip'])
parser.add_argument('mask_cleaning', choices=['cca', 'opening', 'cv_pipeline', 'skeletonize', 'none'])
parser.add_argument('line_extractor', choices=['ransac', 'hough', 'pca'])
args = parser.parse_args()

config_path = Path('configs/evaluation')
for fold in trange(5):
    match args.dataset:
        case 'wrist':
            ds = FractureAngleLineDataset(fold, 'val', True, True)
        case 'hip':
            ds = USHipDataset(fold, 'val', z_std_img=True)

    with open((config_path / args.dataset).with_suffix('.yaml')) as file:
        config = yaml.safe_load(file)
    line_predictor = inference.SegLinePredictor(config['segmentation_model_clearml_task_id_folds'][fold],
                                                args.line_extractor, config['ransac_k'], args.mask_cleaning,
                                                config['opening_radius'] if args.mask_cleaning else config[
                                                    'cv_pipeline_radius'],
                                                config['connected_component_analyse']['min_abs_area'],
                                                config['connected_component_analyse']['min_rel_to_biggest_area'])

    with torch.inference_mode():
        for sample_idx, sample in enumerate(ds):
            x, y, valid = sample['img'], sample['mask'], sample['gt_available']
            y_hat, y_hat_cleaned, (x_coord_hat, y_coord_hat) = line_predictor.predict_for_plotting(x.unsqueeze(0))

            fig, axs = plt.subplots(1, 2)
            for ax in axs.flatten():
                ax.imshow((x * ds.IMG_STD + ds.IMG_MEAN).squeeze(0), 'gray')


            def convert_to_colored_image(mask: torch.Tensor, color: np.ndarray):
                colored = np.zeros((*mask.shape, 4))
                colored[..., :3] = color[:3]  # RGB color
                colored[..., 3] = mask.float().numpy()  # alpha = mask intensity

                return colored


            n_lines = y.shape[0]
            colors = plt.cm.tab10(np.linspace(0, 1, n_lines))
            for line_idx in range(n_lines):
                axs[0].imshow(convert_to_colored_image(y_hat[0, line_idx], colors[line_idx]))
                axs[1].imshow(convert_to_colored_image(y_hat_cleaned[0, line_idx], colors[line_idx]))
                axs[1].plot(x_coord_hat[0, line_idx], y_coord_hat[0, line_idx], 'r')

            fig.savefig(f'/home/ron/Desktop/{args.dataset}/{fold}_{sample_idx}.png')
            plt.close(fig)
