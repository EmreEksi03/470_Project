import openml
import os
import pandas as pd

# Directory to save datasets
save_dir = "openml_cc18_datasets"
os.makedirs(save_dir, exist_ok=True)

# Get the CC18 benchmark suite (ID = 99)
benchmark_suite = openml.study.get_suite(99)  # Corrected method

# Loop through all task IDs in the benchmark
for task_id in benchmark_suite.tasks:
    try:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        df, _, _, _ = dataset.get_data(target=dataset.default_target_attribute)

        # Save as CSV
        file_name = f"{dataset.name}_{dataset.dataset_id}.csv"
        df.to_csv(os.path.join(save_dir, file_name), index=False)
        print(f"✅ Saved: {file_name}")
    except Exception as e:
        print(f"❌ Failed for task {task_id}: {e}")
