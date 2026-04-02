import os

import optuna
from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformIntegerParameterRange
from clearml.automation.optuna import OptimizerOptuna
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('dataset', choices=["wrist", "hip", "autosafe"])
args = parser.parse_args()

task = Task.init('Dislocation/HPO', args.dataset, auto_resource_monitoring=False)
controller = HyperParameterOptimizer(
    base_task_id=dict(hip='eee09e1645dd4d32b572f8db56385103', wrist='97b8e7006a934b679a8d49751be0b9e8',
                      autosafe='31f1bbe7169b4461844fba1160b25b06')[args.dataset],
    hyper_parameters=[
        # lambda
        UniformIntegerParameterRange('Args/fit.model.init_args.blob_sigma', min_value=1, max_value=50, step_size=1),
        UniformIntegerParameterRange('Args/fit.model.init_args.mse_scaling', min_value=2, max_value=100, step_size=2),
    ],
    objective_metric_title='error',
    objective_metric_series='val',
    objective_metric_sign='min_global',
    max_number_of_concurrent_tasks=1,
    optimizer_class=OptimizerOptuna,
    save_top_k_tasks_only=5,
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=150, n_min_trials=3),
    min_iteration_per_job=50,
    max_iteration_per_job=1000,
    total_max_jobs=150,
    pool_period_min=2,
)

controller.start_locally()
# wait until optimization completed or timed-out
controller.wait()
# make sure we stop all jobs
controller.stop()

# save the optuna optimizer results
task.upload_artifact('optuna_study.pkl', artifact_object=controller.get_optimizer()._study, auto_pickle=True)
