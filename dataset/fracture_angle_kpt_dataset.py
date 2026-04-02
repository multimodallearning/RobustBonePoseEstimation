from pathlib import Path
from typing import Any

import pandas as pd
import torch
from kornia import augmentation as A
from pytorch_lightning import LightningDataModule
from torchvision import io
from tqdm import tqdm

from dataset.cvat_parser import CVATBoxParser, LocationCode


class FractureAngleKeyPointDataset(torch.utils.data.Dataset):
    KPT_LBL = [['ulna', 'radius'], ['proximal', 'distal'], ['tl', 'tr', 'br', 'bl']]
    IMG_MEAN = 0.3505533917353781
    IMG_STD = 0.22763733675869177

    def __init__(self, fold: int, mode: str, normalise_laterality: bool = True, z_std_img: bool = False,
                 include_cases_no_gt: bool = False):
        """
        Pytorch dataset for fracture fragment angle estimation in wrist radiographs using landmarks
        :param fold: fold index of cross validation
        :param mode: training or validation
        :param normalise_laterality: if all right hands should be flipped to represent a left one
        :param z_std_img: normalize img to have zero-mean and unit-variance
        :param include_cases_no_gt: if images without ground truth should be excluded
        """
        assert mode in ['train', 'val'], 'Invalid mode'
        # prepare dataset for loading
        s_split = pd.read_csv('dataset/cv_split_fracture_angle.csv', index_col='PatID').squeeze()
        mask = s_split == fold if mode == 'val' else s_split != fold
        available_files = s_split[mask]
        # load data into memory
        if torch.cuda.is_available():  # choosing local SSD of GPU server
            img_path = Path('/data_rechenknecht02_2/keuth/FractureAngle/img8bit')
            print('Using local SSD of GPU Server')
        else:
            img_path = Path('dataset/data/img8bit')
        parser = CVATBoxParser(list(Path('dataset/data/cvat_annotations').glob('*.xml')),
                               [LocationCode.PROXIMAL, LocationCode.DISTAL])
        resize_op = A.AugmentationSequential(A.Resize((384, 224)), data_keys=["image", "keypoints"])
        hflip_op = A.AugmentationSequential(A.RandomHorizontalFlip(p=1.), data_keys=["image", "keypoints"])
        self.data = {}
        img_skipped_cnt = 0
        for img_id in tqdm(available_files.index, desc=f'Processing {mode} samples'):
            # image
            img = io.read_image(img_path / (img_id + '.png')).float().div(255)

            # key points
            visible = torch.zeros(2, 2, dtype=torch.bool)  # [radius, ulna]
            kpts = torch.empty((2, 2, 4, 2), dtype=torch.float)  # [ulna, radius], [proximal, distal], [tl, tr, br, bl]
            try:
                boxes = parser(img_id)
                for box in boxes:
                    visible[box.bone.value, box.location.value] = True
                    kpts[box.bone.value, box.location.value] = box.bbox
            except KeyError:
                # no boxes available
                if not include_cases_no_gt:
                    img_skipped_cnt += 1
                    continue  # skip image

            # flip everything to left
            laterality = img_id.split('-')[1][0]
            if normalise_laterality and laterality == 'R':
                img, kpts = hflip_op(img, kpts.flatten(end_dim=-2))
                kpts = kpts.view(2, 2, 4, 2)
                kpts = kpts[..., [1, 0, 3, 2], :]

            img, kpts = resize_op(img, kpts.flatten(end_dim=-2))
            if z_std_img:
                img = (img - self.IMG_MEAN) / self.IMG_STD

            self.data[img_id] = {
                'img': img.squeeze(0),
                'kpts': kpts.view(2, 2, 4, 2).flip(-1),
                'visible': visible.unsqueeze(-1).repeat_interleave(4, -1),
            }
        self.img_ids = list(self.data.keys())
        if not include_cases_no_gt:
            print(f'Skipped {img_skipped_cnt} due to corrupted or missing annotations.')

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        return self.data[self.img_ids[idx]]

    @staticmethod
    def extract_line_endpoints(kpts: torch.Tensor, visible: torch.Tensor):
        """
        extracts the top and bottom center points of a bounding box representing its center line
        """
        start_pnts = kpts[..., :2, :].mean(-2, keepdim=True)
        end_pnts = kpts[..., 2:, :].mean(-2, keepdim=True)
        kpts_reduced = torch.cat([start_pnts, end_pnts], -2)

        start_pnts_visible = visible[..., :2].all(-1, keepdim=True)
        end_pnts_visible = visible[..., 2:].all(-1, keepdim=True)
        visible_reduced = torch.cat([start_pnts_visible, end_pnts_visible], -1)

        return kpts_reduced, visible_reduced


class FractureAngleKeyPointDataModule(LightningDataModule):
    def __init__(self, fold_idx: int = 0, batch_size: int = 16, use_data_aug: bool = True,
                 box_center_line_kpts: bool = True):
        """
        :param fold_idx: fold index of cross validation
        :param batch_size: used batch size
        :param use_data_aug: use data augmentation on training data
        :param box_center_line_kpts: if to use all four bounding box corners or use two representing its center line
        """
        super().__init__()
        self.fold = fold_idx
        self.center_box_kpts = box_center_line_kpts
        self.dl_kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
        self.data_aug = A.AugmentationSequential(A.RandomHorizontalFlip(p=0.5),
                                                 A.RandomAffine(15, (0.05,) * 2, (0.9, 1.1)),
                                                 data_keys=["input", "keypoints"])
        self.data_aug = self.data_aug if use_data_aug else None

    def setup(self, stage: str) -> None:
        self.val_ds = FractureAngleKeyPointDataset(self.fold, 'val', True, True)
        if stage == 'fit':
            self.train_ds = FractureAngleKeyPointDataset(self.fold, 'train', True, True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, drop_last=True, **self.dl_kwargs)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, drop_last=False, **self.dl_kwargs)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, drop_last=False, **self.dl_kwargs)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.center_box_kpts:
            batch['kpts'], batch['visible'] = FractureAngleKeyPointDataset.extract_line_endpoints(batch['kpts'],
                                                                                                  batch['visible'])
        # flatten keypoints
        batch['kpts'] = batch['kpts'].flatten(start_dim=1, end_dim=-2)
        batch['visible'] = batch['visible'].flatten(start_dim=1)

        return batch

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        trainer = getattr(self, "trainer", None)
        if self.data_aug and trainer and self.trainer.training:
            batch['img'], batch['kpts'] = self.data_aug(batch['img'], batch['kpts'])

        return batch


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    ds = FractureAngleKeyPointDataset(0, 'val')
    sample = ds[0]
    plt.imshow(sample['img'].squeeze())
    kpts, vis = ds.extract_line_endpoints(sample['kpts'], sample['visible'])
    kpts = kpts.flatten(end_dim=-2)
    plt.scatter(kpts[:, 1], kpts[:, 0])
    # plot indices
    for i, (y, x) in enumerate(kpts):
        plt.text(x, y, str(i), color='yellow', fontsize=10, ha='center', va='center')
    plt.show()