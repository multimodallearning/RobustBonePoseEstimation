from pathlib import Path
from typing import Any

import pandas as pd
import torch
from kornia import augmentation as A
from kornia.morphology import dilation
from numpy import clip
from pytorch_lightning import LightningDataModule
from skimage import draw, morphology
from torchvision import io
from tqdm import tqdm

from dataset.cvat_parser import CVATBoxParser, LocationCode


class FractureAngleLineDataset(torch.utils.data.Dataset):
    SEG_MASK_LBL = [['ulna', 'radius'], ['proximal', 'distal']]
    IMG_MEAN = 0.3505533917353781
    IMG_STD = 0.22763733675869177

    def __init__(self, fold: int, mode: str, normalise_laterality: bool = True, z_std_img: bool = False,
                 include_cases_no_gt: bool = False, dilation_disk_radius: int = 2):
        """
        Pytorch dataset for fracture fragment angle estimation in wrist radiographs
        :param fold: fold index of cross validation
        :param mode: training or validation
        :param normalise_laterality: if all right hands should be flipped to represent a left one
        :param z_std_img: normalize img to have zero-mean and unit-variance
        :param include_cases_no_gt: if images without ground truth should be excluded
        :param dilation_disk_radius: controlling the dilation to widen the one-pixel ground truth line
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
        resize_op = A.AugmentationSequential(A.Resize((384, 224)), data_keys=["image", "image"])
        hflip_op = A.AugmentationSequential(A.RandomHorizontalFlip(p=1.), data_keys=["image", "mask"])
        dilation_kernel = torch.from_numpy(morphology.disk(dilation_disk_radius, strict_radius=False)).float()
        self.data = {}
        img_skipped_cnt = 0
        for img_id in tqdm(available_files.index, desc=f'Processing {mode} samples'):
            # image
            img = io.read_image(img_path / (img_id + '.png')).float().div(255)

            # key points
            seg_mask = torch.zeros((2, 2, *img.shape[1:]))
            try:
                boxes = parser(img_id)
                for box in boxes:
                    line_start = box.bbox[:2].mean(0).round().int()
                    line_end = box.bbox[2:].mean(0).round().int()
                    rr, cc, val = draw.line_aa(line_start[1], line_start[0], line_end[1], line_end[0])
                    rr, cc = clip(rr, 0, seg_mask.shape[-2] - 1), clip(cc, 0, seg_mask.shape[-1] - 1)
                    seg_mask[box.bone.value, box.location.value, rr, cc] = torch.from_numpy(val).float()
            except KeyError:
                # no boxes available
                if not include_cases_no_gt:
                    img_skipped_cnt += 1
                    continue  # skip image
            seg_mask = seg_mask.flatten(end_dim=1)
            img, seg_mask = resize_op(img, seg_mask)
            # dilate segmentation
            seg_mask = dilation(seg_mask, dilation_kernel, engine='convolution')
            seg_mask = seg_mask > 0

            # flip everything to left
            laterality = img_id.split('-')[1][0]
            if normalise_laterality and laterality == 'R':
                img, seg_mask = hflip_op(img, seg_mask)
            # normalize
            if z_std_img:
                img = (img - self.IMG_MEAN) / self.IMG_STD

            self.data[img_id] = {
                'img': img.squeeze(0),
                'mask': seg_mask.squeeze(0),
                'gt_available': seg_mask.flatten(start_dim=2).any(-1).squeeze(0)
            }
        self.img_ids = list(self.data.keys())
        if not include_cases_no_gt:
            print(f'Skipped {img_skipped_cnt} due to corrupted or missing annotations.')

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        return self.data[self.img_ids[idx]]


class FractureAngleLineDataModule(LightningDataModule):
    def __init__(self, fold_idx: int = 0, batch_size: int = 32, use_data_aug: bool = True):
        """
        :param fold_idx: fold index of cross validation
        :param batch_size: used batch size
        :param use_data_aug: use data augmentation on training data
        """
        super().__init__()
        self.fold = fold_idx
        self.dl_kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
        self.data_aug = A.AugmentationSequential(A.RandomHorizontalFlip(p=0.5),
                                                 A.RandomAffine(15, (0.05,) * 2, (0.9, 1.1)),
                                                 data_keys=["input", "mask"])
        self.data_aug = self.data_aug if use_data_aug else None

    def setup(self, stage: str) -> None:
        self.val_ds = FractureAngleLineDataset(self.fold, 'val', True, True)
        if stage == 'fit':
            self.train_ds = FractureAngleLineDataset(self.fold, 'train', True, True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, drop_last=True, **self.dl_kwargs)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, drop_last=False, **self.dl_kwargs)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, drop_last=False, **self.dl_kwargs)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        trainer = getattr(self, "trainer", None)
        if self.data_aug and trainer and self.trainer.training:
            batch['img'], batch['mask'] = self.data_aug(batch['img'], batch['mask'])

        return batch


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    ds = FractureAngleLineDataset(0, 'val')
    for idx, sample in enumerate(ds):
        img = sample['img'].squeeze()
        masks = sample['mask']

        fig, axs = plt.subplots(1, 5)
        # fig.suptitle(ds.img_ids[idx])
        axs[0].imshow(img)
        for i, mask in enumerate(masks, start=1):
            axs[i].imshow(img, 'gray')
            axs[i].imshow(mask.squeeze(0), alpha=.5)

        fig.tight_layout()
        fig.savefig(f'/home/ron/Desktop/graz_line/{ds.img_ids[idx]}.png', bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)
