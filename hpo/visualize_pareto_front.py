import optuna
from optuna import visualization
from plotly.io import show
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('dataset', choices=['wrist', 'hip', 'autosafe'])
dataset = parser.parse_args().dataset

study = optuna.create_study(storage=f"sqlite:///hpo/{dataset}.sqlite3", study_name=f"{dataset}_smart_ccv",
                            load_if_exists=True)
fig = visualization.plot_pareto_front(study, target_names=['L1 angle', 'ratio valid'])
show(fig)
