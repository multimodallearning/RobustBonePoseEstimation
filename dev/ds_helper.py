from dataset.fracture_angle_dataset import FractureAngleKeyPointDataset
from torch.utils.data import ConcatDataset

def get_ds_including_all_data():
    ds_kwargs = dict(fold=0, normalise_laterality=True, z_std_img=False, include_cases_no_gt=False)
    ds = ConcatDataset([
        FractureAngleKeyPointDataset(mode='train', **ds_kwargs),
        FractureAngleKeyPointDataset(mode='val', **ds_kwargs)
    ])
    return ds