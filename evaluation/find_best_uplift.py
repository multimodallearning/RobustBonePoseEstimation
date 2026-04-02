# script to find the sample with the biggest error difference

import pandas as pd
from argparse import ArgumentParser
from numpy import genfromtxt

parser = ArgumentParser()
parser.add_argument('first_method', type=str)
parser.add_argument('second_method', type=str)
args = parser.parse_args()

first_id = args.first_method
second_id = args.second_method
assert first_id.split('_')[0] == second_id.split('_')[0], 'dataset mismatch'
results_first = genfromtxt(f'evaluation/l1_errors/{first_id}.csv', delimiter=',')
results_second = genfromtxt(f'evaluation/l1_errors/{second_id}.csv', delimiter=',')
df_idx = pd.read_csv(f'evaluation/fold_sample_idx_mapping/{first_id.split('_')[0]}.csv')
assert len(results_first) == len(results_second) == len(df_idx)
df = df_idx
df['delta'] = abs(results_second - results_first)
pass