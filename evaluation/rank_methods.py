# script that iterate over all results and rank the methods based on significant tests
from pathlib import Path

import pandas as pd
import torch
from scipy.stats import gmean

from evaluation import rank_utils
from numpy import genfromtxt

# construct metric matrix
path_error_files = Path('evaluation/l1_errors')
datasets = {f.stem.split('_')[0] for f in path_error_files.glob('*.csv')}
df = dict()
df_normalize = dict()
for ds in sorted(datasets):
    l1_errors = list()
    method_names = list()
    for csv_file in sorted(list(path_error_files.glob(f'{ds}_*.csv'))):
        l1_errors.append(torch.from_numpy(genfromtxt(csv_file, delimiter=',')))
        method_names.append(csv_file.stem.split('_', 1)[1])
    l1_errors = torch.stack(l1_errors)
    # ranking
    ranking_scores, better = rank_utils.has_lower_error(l1_errors, 'signed-rank')
    df[ds] = pd.Series(index=method_names, data=ranking_scores.tolist())
    ranking_scores_normalized = rank_utils.rankscore_avgtie(ranking_scores)
    df_normalize[ds] = pd.Series(index=method_names, data=ranking_scores_normalized.tolist())
    df_is_better = pd.DataFrame(index=method_names, columns=method_names, data=better.bool())
    print('\n\n', ds)
    print(df_is_better.to_string())

df = pd.DataFrame(df)
print(df.to_string())

df_normalize = pd.DataFrame(df_normalize)
df_normalize['gmean'] = df_normalize.apply(lambda row: gmean(row.dropna()), 1)
df_normalize.sort_values(by='gmean', ascending=False, inplace=True)
print(df_normalize.to_string())
