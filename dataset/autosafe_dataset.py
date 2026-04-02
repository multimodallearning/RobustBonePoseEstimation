from functools import reduce
from pathlib import Path
from random import shuffle, seed
from typing import Any

import albumentations as A
import cv2
import pandas as pd
import torch
from kornia import augmentation as K
from pytorch_lightning import LightningDataModule
from torch.nn import functional as F
from tqdm import tqdm


class AutoSafeDataset(torch.utils.data.Dataset):
    IMG_MEAN = 0.1143
    IMG_STD = 0.1562

    def __init__(self, fold: int, mode: str, z_std_img: bool = False):
        """
        Pytorch dataset for autosafe data (fracture fragment angle estimation in ultrasound)
        :param fold: fold index of cross validation
        :param mode: training or validation
        :param z_std_img: normalize img to have zero-mean and unit-variance
        """
        assert mode in ['train', 'val', 'all'], 'Invalid mode'
        if torch.cuda.is_available():
            print('Use GPU server SSD')
            data_dir = Path('/data_rechenknecht02_2/keuth/FractureAngle/AUTOSAFE_angle_detection_dataset')
        else:
            data_dir = Path('dataset/data/AUTOSAFE_angle_detection_dataset')
        df_kpts = pd.read_csv(data_dir / 'keypoints.csv', index_col=0)

        # create cross validation folds (5 folds)
        cases_ids = sorted(df_kpts.index.to_list())
        seed(42)
        shuffle(cases_ids)
        n = len(cases_ids)
        cases_ids_folds = [cases_ids[i * n // 5:(i + 1) * n // 5] for i in range(5)]
        if mode == 'train':
            cases_ids_folds.pop(fold)
            selected_case_ids = reduce(lambda x, y: x + y, cases_ids_folds)
        elif mode == 'val':
            selected_case_ids = cases_ids_folds.pop(fold)
        elif mode == 'all':
            selected_case_ids = cases_ids
        else:
            raise ValueError('Unknown mode. Use train, val or all')

        resize_op = A.Compose([
            A.LongestMaxSize(max_size=320, area_for_downscale="image"),
            A.PadIfNeeded(min_height=320, min_width=320),
            A.ToFloat(255),
            A.ToTensorV2()
        ], keypoint_params=A.KeypointParams('yx', remove_invisible=False))
        self.data = list()
        for case_id in tqdm(selected_case_ids, desc=f'Loading {mode} data'):
            img = cv2.cvtColor(cv2.imread(data_dir / 'images' / (case_id + '.png')), cv2.COLOR_BGR2GRAY)
            seg_mask = cv2.cvtColor(cv2.imread(data_dir / 'prepared_masks' / (case_id + '.png')), cv2.COLOR_BGR2GRAY)
            kpts = df_kpts.loc[case_id].to_numpy().reshape(4, 2)
            # resize
            resized = resize_op(image=img, keypoints=kpts, mask=seg_mask)

            # one hot encoding and drop background
            seg_mask = F.one_hot(resized['mask'].long(), 3).float().movedim(-1, 0)[1:].bool()

            # normalize
            img = resized['image']
            if z_std_img:
                img = (img - self.IMG_MEAN) / self.IMG_STD

            self.data.append({
                'img': img,
                'mask': seg_mask,
                'gt_available': torch.ones(2, dtype=torch.bool),
                'kpts': resized['keypoints'],
                'visible': torch.ones(4, dtype=torch.bool)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AutoSafeDataModule(LightningDataModule):
    def __init__(self, fold_idx: int = 0, batch_size: int = 16, use_data_aug: bool = True):
        """
        :param fold_idx: fold index of cross validation
        :param batch_size: used batch size
        :param use_data_aug: use data augmentation on training data
        """
        super().__init__()
        self.fold = fold_idx
        self.dl_kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
        self.data_aug = K.AugmentationSequential(K.RandomAffine(15, (0.05,) * 2, (0.9, 1.1), p=0.8),
                                                 data_keys=["input", "mask", "keypoints"])
        self.data_aug = self.data_aug if use_data_aug else None

    def setup(self, stage: str) -> None:
        self.val_ds = AutoSafeDataset(self.fold, 'val', True)
        if stage == 'fit':
            self.train_ds = AutoSafeDataset(self.fold, 'train', True)

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

    ds = AutoSafeDataset(0, 'all')
    for idx, sample in enumerate(ds):
        img = sample['img'].squeeze()
        masks = sample['mask']
        kpts = sample['kpts']

        fig, axs = plt.subplots(1, 3)
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
