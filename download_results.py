import neptune
import pandas as pd

project = neptune.init_project(
    mode="read-only"
)

runs = project.fetch_runs_table().to_pandas()
all_series_data = []

run_ids = runs['sys/id'].tolist()

print(run_ids)

for i, run_id in enumerate(run_ids):
    specific_run = neptune.init_run(with_id=run_id, mode="read-only")
    tags = specific_run["sys/tags"].fetch()

    training_metrics = specific_run.get_structure()["training"]

    for metric in training_metrics:
        if 'FloatSeries' in str(type(training_metrics[metric])):
            series_data = specific_run["training"][metric].fetch_values()
            series_df = pd.DataFrame(series_data)
            series_df['namespace'] = "training"
            series_df['metric'] = metric
            series_df['run_id'] = run_id
            series_df['tags'] = [list(tags.copy()) for _ in range(len(series_df))]
            for hparam, hparam_value in specific_run["training"]["hyperparams"].fetch().items():
                series_df[hparam] = hparam_value
            all_series_data.append(series_df)

    specific_run.stop()

pd.concat(all_series_data).to_csv("series_data.csv", index=False)
