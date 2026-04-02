# script to create five folded cross validation split grouped over patient id

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from dataset.cvat_parser import CVATBoxParser, LocationCode
from pathlib import Path

xml_files = list(Path('dataset/data/cvat_annotations').glob('*.xml'))
parser = CVATBoxParser(xml_files, [LocationCode.PROXIMAL, LocationCode.DISTAL])
available_files = list(parser.boxes.keys())
df = pd.read_csv('dataset/dataset.csv', index_col='filestem')
df_filtered = df[df.index.isin(map(lambda s: s.split('.')[0], available_files)) & df['projection'].isin([1, 2])]
x = df_filtered.index
y = df_filtered['projection']
g = df_filtered['patient_id']

split = pd.Series(index=x)
for i, (train_idx, test_idx) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x, y, g)):
    print(f"Fold {i}:")
    print("Train:", len(train_idx), "Test:", len(test_idx))
    split[x[test_idx]] = i
split = split.astype(int)
assert not any(split.isna())
split.to_csv('dataset/cv_split_fracture_angle.csv', index=True, header=['ValSplitIndex'], index_label='PatID')