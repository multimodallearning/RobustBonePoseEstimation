# script to create csv mapping the index of all data to their cross validation split index and its relative index to it
from dataset import fracture_angle_line_dataset, us_hip_dataset, autosafe_dataset
import pandas as pd

for ds_name in ['autosafe', 'wrist', 'hip']:
    fold_idx = []
    sample_idx = []
    for fold in range(1, 5): # skipping 0 since it was used for HPO
        match ds_name:
            case 'wrist':
                ds = fracture_angle_line_dataset.FractureAngleLineDataset(fold, 'val')
            case 'hip':
                ds = us_hip_dataset.USHipDataset(fold, 'val')
            case 'autosafe':
                ds = autosafe_dataset.AutoSafeDataset(fold, 'val')

        fold_idx.extend([fold] * len(ds))
        sample_idx.extend(list(range(len(ds))))
    df = pd.DataFrame(dict(fold=fold_idx, sample=sample_idx))
    df.to_csv(f'evaluation/fold_sample_idx_mapping/{ds_name}.csv', index=False)
