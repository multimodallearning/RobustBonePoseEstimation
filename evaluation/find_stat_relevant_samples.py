# script to find the index of best, worst and median sample

import pandas as pd
from argparse import ArgumentParser
from numpy import genfromtxt

parser = ArgumentParser()
parser.add_argument('method', type=str)
args = parser.parse_args()

results = genfromtxt(f'evaluation/l1_errors/{args.method}.csv', delimiter=',')
df_idx = pd.read_csv(f'evaluation/fold_sample_idx_mapping/{args.method.split('_')[0]}.csv')
df = df_idx
df['error'] = results
df['diff_median'] = (df['error'] - df['error'].median()).abs()
print('best\n', df.loc[df['error'].idxmin()])
print('median\n', df.loc[(df['error'] - df['error'].median()).abs().idxmin()])
print('worst\n', df.loc[df['error'].idxmax()])
