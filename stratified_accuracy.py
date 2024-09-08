import numpy as np
import pandas as pd
import json

test_results = pd.read_csv('test_results.csv')

test_results["ground_truth"] = test_results["label"]
test_results["prediction"] = test_results.apply(lambda row: np.argmax(json.loads(row["probability"])), axis=1)
del test_results["probability"]

test_results['label'] = test_results['label'].replace({0: 'Not Wavy', 1: 'Wavy'})
test_results['attribute'] = test_results['attribute'].replace({0: 'Female', 1: 'Male'})

accuracy = test_results.groupby(['label', 'attribute']).apply(
    lambda group: (group['prediction'] == group['ground_truth']).mean()
)

counts = test_results.groupby(['label', 'attribute']).size().reset_index(name='count')

print(accuracy)
print(counts)