from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml

from dataset.fracture_angle_line_dataset import FractureAngleLineDataModule
from dataset.us_hip_dataset import USHipDataModule
from dataset.autosafe_dataset import AutoSafeDataModule
from utils import eval_utils
from utils import inference
import csv

parser = ArgumentParser()
parser.add_argument('dataset', choices=['wrist', 'hip', 'autosafe'])
parser.add_argument('mask_cleaning', choices=['cca', 'opening', 'cv_pipeline', 'skeletonize', 'none'])
parser.add_argument('line_extractor', choices=['ransac', 'hough', 'pca'])
args = parser.parse_args()

config_path = Path('configs/evaluation')
error = []
fold_and_index = []
for fold in range(1, 5): # skipping 0 since it was used for HPO
    match args.dataset:
        case 'wrist':
            dm = FractureAngleLineDataModule(fold, 1)
        case 'hip':
            dm = USHipDataModule(fold, 1)
        case 'autosafe':
            dm = AutoSafeDataModule(fold, 1)

    dm.setup('test')
    yaml_file = config_path / args.dataset
    with open(yaml_file.with_suffix('.yaml')) as file:
        config = yaml.safe_load(file)
    ransac_k = config['ransac_k_skeletonize'] if args.mask_cleaning == 'skeletonize' else config['ransac_k']
    line_predictor = inference.SegLinePredictor(config['segmentation_model_clearml_task_id_folds'][fold],
                                                args.line_extractor, ransac_k, args.mask_cleaning,
                                                config['opening_radius'] if args.mask_cleaning else config[
                                                    'cv_pipeline_radius'],
                                                config['connected_component_analyse']['min_abs_area'],
                                                config['connected_component_analyse']['min_rel_to_biggest_area'])
    line_model = line_predictor.pca_line_model

    v_hat = []
    v = []
    with torch.inference_mode():
        for i, batch in enumerate(dm.test_dataloader()):
            x, y, valid = batch['img'], batch['mask'], batch['gt_available']
            current_y_hat = line_predictor.predict_masks(x)
            v_hat = line_predictor.predict(x)[0]
            v = line_model.estimate(y)[0]
            alpha_hat, beta_hat = eval_utils.extract_alpha_beta(v_hat, args.dataset)
            alpha, beta = eval_utils.extract_alpha_beta(v, args.dataset)
            l1 = torch.tensor([alpha - alpha_hat, beta - beta_hat]).abs().nanmean()
            error.append(l1)
            fold_and_index.append([fold, i])

error = torch.stack(error)
fold_and_index = torch.tensor(fold_and_index)
print(error.nanmean(), error[~error.isnan()].std())
best_idx = fold_and_index[error.nan_to_num(nan=torch.inf).argmin()]
print(f'best sample: fold {best_idx[0]}, sample {best_idx[1]} with {error.nan_to_num(nan=torch.inf).min().item()}')
median_idx = fold_and_index[error == error.nanmedian()].squeeze(0)
print(f'median sample: fold {median_idx[0]}, sample {median_idx[1]} with {error.nanmedian()}')
worst_idx = fold_and_index[error.nan_to_num(nan=-torch.inf).argmax()]
print(f'worst sample: fold {worst_idx[0]}, sample {worst_idx[1]} with {error.nan_to_num(nan=-torch.inf).max().item()}')

with open(f"evaluation/l1_errors/{args.dataset}_{args.mask_cleaning}_{args.line_extractor}.csv", 'x') as f:
    writer = csv.writer(f)
    writer.writerow(error.tolist())
