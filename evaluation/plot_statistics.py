# script that create the tables and plots for each dataset
import pandas as pd
import seaborn as sns
from pathlib import Path
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

parser = ArgumentParser()
parser.add_argument('dataset', choices=['wrist', 'hip', 'autosafe'])
dataset = parser.parse_args().dataset

match dataset:
    case 'wrist':
        axins_x_lim = (6, 17)
        axins_y_lim = (0.75, 0.99)
    case 'hip':
        axins_x_lim = (7, 14)
        axins_y_lim = (0.75, 0.99)
    case 'autosafe':
        axins_x_lim = (6, 22)
        axins_y_lim = (0.75, 0.98)

data_list = []

for csv_file in Path('evaluation/l1_errors').glob(f'{dataset}_*.csv'):
    series = pd.read_csv(csv_file, header=None).squeeze()

    # Extract metadata from filename
    parts = csv_file.stem.split('_')
    mask_cleaning = parts[1]
    line_predictor = parts[-1]

    # Expand series values into individual rows
    for val in series.values:
        data_list.append({
            "mask_cleaning": mask_cleaning,
            "line_predictor": line_predictor,
            "l1_error": val
        })

df_long = pd.DataFrame(data_list)
stats_per_group = df_long.groupby(['line_predictor', 'mask_cleaning']).agg(['mean', 'std', 'median']).round(2)
print(stats_per_group)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_long, x='line_predictor', y='l1_error', hue='mask_cleaning', showfliers=False)
plt.title(f"{dataset} L1 Errors by Line Predictor")

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(5, 4))
df_ecdf = df_long
df_ecdf = df_ecdf[~df_ecdf['mask_cleaning'].isin(['opening', 'skeletonize'])]
df_ecdf["method"] = df_ecdf["mask_cleaning"] + " | " + df_ecdf["line_predictor"]
df_ecdf.sort_values(by=['method'], inplace=True)
sns.ecdfplot(data=df_ecdf, x="l1_error", hue="method", legend=False, ax=ax)
ax.set_xlabel("L1 Error [°]")
ax.set_ylabel("Cumulative Probability")

# Create a zoomed inset
axins = inset_axes(ax, width="50%", height="50%", loc='lower right')
sns.ecdfplot(data=df_ecdf, x="l1_error", hue="method", legend=False, ax=axins)
axins.set_xlim(axins_x_lim)
axins.set_ylim(axins_y_lim)
axins.set_xticks([])
axins.set_yticks([])
axins.set_xlabel('')
axins.set_ylabel('')
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", lw=1)

plt.savefig(f'/home/ron/Documents/Konferenzen/MIAU/ecdf_{dataset}.pdf',  bbox_inches='tight', pad_inches=0)

plt.show()
