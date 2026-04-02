# prepare autosafe mask for training:
#   * only extract bone segmentation with at least two instances
#   * if more then two instances → discard all but the two largest
#   + for each instance extract the most left and right coordinates as landmarks to represent start and end of line

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops

base_dir = Path('dataset/data/AUTOSAFE_angle_detection_dataset')
label_of_interest = 1
df_kpts = dict()

for mask_file in (base_dir / 'masks').glob('*.png'):
    mask = Image.open(mask_file).convert('L')
    mask = np.asarray(mask) == label_of_interest
    cc_img, n_regions = label(mask, return_num=True)
    if n_regions < 2:
        print('Missing segmentation in', mask_file.stem)
        continue

    regions = regionprops(cc_img, cache=True)
    # selecting the two biggest components
    regions.sort(key=lambda r: r.area, reverse=True)
    regions = regions[:2]
    # sort from left to right
    regions.sort(key=lambda r: r.centroid[1])
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    kpts = np.empty((2, 2, 2), dtype=np.uint16)
    for i, props in enumerate(regions):
        rows, cols = props.coords.T
        new_mask[rows, cols] = i + 1

        kpts[i, 0] = props.coords[np.argmin(props.coords[:, 1])]
        kpts[i, 1] = props.coords[np.argmax(props.coords[:, 1])]

    Image.fromarray(new_mask).save(base_dir / 'prepared_masks' / mask_file.name)
    df_kpts[mask_file.stem] = kpts.flatten()

column_labels = ['_'.join([p, i, c]) for i, p, c in product(["1", "2"], ['start', 'end'], ['x', 'y'])]
df_kpts = pd.DataFrame.from_dict(df_kpts, orient='index', columns=column_labels)
df_kpts.to_csv(base_dir / 'keypoints.csv')
