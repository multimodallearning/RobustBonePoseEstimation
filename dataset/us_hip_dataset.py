from functools import reduce
from pathlib import Path
from random import shuffle, seed
from typing import Any

import pandas
import torch
from kornia import augmentation as A
from kornia.morphology import dilation
from pytorch_lightning import LightningDataModule
from skimage import morphology
from torchvision import io
from tqdm import tqdm


class USHipDataset(torch.utils.data.Dataset):
    IMG_MEAN = 0.1507
    IMG_STD = 0.1476

    def __init__(self, fold: int, mode: str, z_std_img: bool = False, dilation_disk_radius: int = 2):
        """
        Pytorch dataset for DDH in ultrasound
        :param fold: fold index of cross validation
        :param mode: training or validation
        :param z_std_img: normalize img to have zero-mean and unit-variance
        :param dilation_disk_radius: controlling the dilation to widen the one-pixel ground truth line
        """
        assert mode in ['train', 'val', 'all'], 'Invalid mode'
        dilation_kernel = torch.from_numpy(morphology.disk(dilation_disk_radius, strict_radius=False)).float()

        if torch.cuda.is_available():
            print('Use GPU server SSD')
            data_dir = Path('/data_rechenknecht02_2/keuth/FractureAngle/data_us_hip_line_seg')
        else:
            data_dir = Path('dataset/data/data_us_hip_line_seg')

        # create cross validation folds (5 folds)
        cases_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        seed(42)
        shuffle(cases_dirs)
        n = len(cases_dirs)
        cases_dirs_folds = [cases_dirs[i * n // 5:(i + 1) * n // 5] for i in range(5)]
        if mode == 'train':
            cases_dirs_folds.pop(fold)
            selected_case_dirs = reduce(lambda x, y: x + y, cases_dirs_folds)
        elif mode == 'val':
            selected_case_dirs = cases_dirs_folds.pop(fold)
        elif mode == 'all':
            selected_case_dirs = cases_dirs
        else:
            raise ValueError('Unknown mode. Use train, val or all')

        # load key points
        df_kpts = pandas.read_csv(data_dir / 'landmarks.csv')
        kpts = torch.from_numpy(df_kpts.drop(columns=['img']).to_numpy()).float()
        kpts = kpts.reshape(len(df_kpts), -1, 2).flip(-1)

        self.data = list()
        for case_path in tqdm(selected_case_dirs, desc=f'Loading {mode} data'):
            case_no = case_path.name
            img = io.read_image(case_path / f'img_{case_no}.png').float().div(255)
            seg_mask = torch.cat([
                io.read_image(data_dir / case_no / f'seg_{case_no}_{i}.png') for i in range(3)
            ]).float().unsqueeze(0)
            seg_mask = dilation(seg_mask, dilation_kernel, engine='convolution')
            seg_mask = seg_mask > 0

            # normalize
            if z_std_img:
                img = (img - self.IMG_MEAN) / self.IMG_STD

            self.data.append({
                'img': img,
                'mask': seg_mask.squeeze(0),
                'gt_available': seg_mask.flatten(start_dim=2).any(-1).squeeze(0),
                'kpts': kpts[int(case_no)],
                'visible': torch.ones(kpts.shape[1], dtype=torch.bool)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class USHipDataModule(LightningDataModule):
    def __init__(self, fold_idx: int = 0, batch_size: int = 16, use_data_aug: bool = True):
        """
        :param fold_idx: fold index of cross validation
        :param batch_size: used batch size
        :param use_data_aug: use data augmentation on training data
        """
        super().__init__()
        self.fold = fold_idx
        self.dl_kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
        self.data_aug = A.AugmentationSequential(A.RandomAffine(15, (0.05,) * 2, (0.9, 1.1), p=0.8),
                                                 data_keys=["input", "mask", "keypoints"])
        self.data_aug = self.data_aug if use_data_aug else None

    def setup(self, stage: str) -> None:
        self.val_ds = USHipDataset(self.fold, 'val', True)
        if stage == 'fit':
            self.train_ds = USHipDataset(self.fold, 'train', True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, drop_last=True, **self.dl_kwargs)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, drop_last=False, **self.dl_kwargs)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, drop_last=False, **self.dl_kwargs)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        trainer = getattr(self, "trainer", None)
        if self.data_aug and trainer and self.trainer.training:
            batch['img'], batch['mask'], batch['kpts'] = self.data_aug(batch['img'], batch['mask'], batch['kpts'])

        return batch


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    ds = USHipDataset(0, 'val')
    for idx, sample in enumerate(ds):
        img = sample['img'].squeeze()
        masks = sample['mask']
        kpts = sample['kpts']

        fig, axs = plt.subplots(1, 4)
        # fig.suptitle(ds.img_ids[idx])
        axs[0].imshow(img)
        for i, mask in enumerate(masks, start=1):
            axs[i].imshow(img, 'gray')
            axs[i].imshow(mask.squeeze(0), alpha=.5)
            axs[i].scatter(kpts[:, 1], kpts[:, 0])
            for j, (y, x) in enumerate(kpts):
                axs[i].text(x, y, str(j), color='yellow', fontsize=10, ha='center', va='center')

        fig.tight_layout()
        # fig.savefig(f'/home/ron/Desktop/hip_line/{idx}.png', bbox_inches="tight", pad_inches=0, dpi=300)
        # plt.close(fig)
        plt.show()
