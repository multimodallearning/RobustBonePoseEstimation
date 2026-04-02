from dataset import fracture_angle_line_dataset, us_hip_line_dataset
from argparse import ArgumentParser
from utils.ransac_util import PCALineModel
import torch
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt

parser = ArgumentParser()
parser.add_argument('dataset', choices=['wrist', 'hip'])
ds_choice = parser.parse_args().dataset

if ds_choice == 'wrist':
    ds = fracture_angle_line_dataset.FractureAngleLineDataset(0, 'train', False)
    hw = (384, 224)
    N = 4
else:
    ds = us_hip_line_dataset.USHipLineDataset(0, 'all')
    hw = (384, 320)
    N = 3

line_model = PCALineModel(hw)

residuals = torch.empty(len(ds), N)
for i, sample in enumerate(tqdm(ds, desc='Calculating residuals')):
    masks = sample['mask']
    _, _, res = line_model.estimate(masks.unsqueeze(0), True)
    residuals[i] = res

sns.histplot(residuals)
plt.show()

