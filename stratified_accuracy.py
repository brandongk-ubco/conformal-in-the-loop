import numpy as np
import pandas as pd
import json
import seaborn as sns
from matplotlib import pyplot as plt

quantile = 0.38182199001312256

test_results = pd.read_csv('test_results.csv')

test_results["ground_truth"] = test_results["label"]
test_results["probability"] = test_results.apply(lambda row: np.array(json.loads(row["probability"])), axis=1)
test_results["prediction"] = test_results.apply(lambda row: np.argmax(row["probability"]), axis=1)
test_results["prediction_set"] = test_results.apply(lambda row: np.array([int(r > quantile) for r in row["probability"]]), axis=1)
test_results["set_size"] = test_results.apply(lambda row: row["prediction_set"].sum(), axis=1)

test_results['label'] = test_results['label'].replace({0: 'Not Wavy', 1: 'Wavy'})
test_results['attribute'] = test_results['attribute'].replace({0: 'Female', 1: 'Male'})

accuracy = test_results.groupby(['label', 'attribute']).apply(
    lambda group: (group['prediction'] == group['ground_truth']).mean()
)

counts = test_results.groupby(['label', 'attribute']).size().reset_index(name='count')

test_results["realized"] = test_results.apply(lambda row: row["prediction_set"][row["ground_truth"]] == 1 and row["set_size"] == 1 , axis=1)
test_results["uncertain"] = test_results.apply(lambda row: row["set_size"] > 1 , axis=1)
test_results["confused"] = test_results.apply(lambda row: row["prediction_set"][row["ground_truth"]] == 0 and row["set_size"] == 1 , axis=1)
test_results["atypical"] = test_results.apply(lambda row: row["set_size"] == 0 , axis=1)

test_results["result"] = test_results.apply(lambda row: "realized" if row["realized"] else "uncertain" if row["uncertain"] else "confused" if row["confused"] else "atypical", axis=1)

breakdown = test_results.groupby(['result']).size().reset_index(name='count')

# grouped_counts = test_results.groupby(["result", 'label', 'attribute']).size().reset_index(name='count')


confused_results = test_results[test_results["confused"]]
confused_counts = confused_results.groupby(['label', 'attribute']).size().reset_index(name='count')

uncertain_results = test_results[test_results["uncertain"]]
uncertain_counts = uncertain_results.groupby(['label', 'attribute']).size().reset_index(name='count')

print(breakdown)
print(accuracy)
print(counts)
print(confused_counts)
print(uncertain_counts)

joined_counts = counts.merge(confused_counts, on=["label", "attribute"], how="outer").fillna(0)
joined_counts = joined_counts.rename(columns={"count_x": "count", "count_y": "confused_count"})
joined_counts["confused_percent"] = joined_counts["confused_count"] / joined_counts["count"]

confused_counts["percent"] = confused_counts["count"] / confused_counts["count"].sum()
counts["percent"] = counts["count"] / counts["count"].sum()

del confused_counts["count"]
del counts["count"]

print("Overall Distribution")
print(counts)

print("Confused Distribution")
print(confused_counts)

test_results["probability_range"] = test_results.apply(lambda row: row["probability"].max() - row["probability"].min(), axis=1)


sns.boxplot(y=test_results["probability_range"], x=test_results["label"], hue=test_results["attribute"])
plt.savefig("probability_range.png")
