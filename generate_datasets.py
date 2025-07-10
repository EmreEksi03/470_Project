import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.utils import shuffle

def generate_and_save_datasets(
    output_folder="generated_datasets",
    n_datasets=50,
    random_seed=42
):
    np.random.seed(random_seed)
    os.makedirs(output_folder, exist_ok=True)

    for i in range(n_datasets):
        # Random configuration
        n_samples = np.random.randint(50, 1000)
        n_features = np.random.randint(5, 50)
        n_informative = np.random.randint(2, n_features)
        n_redundant = np.random.randint(0, n_features - n_informative)
        n_classes = np.random.randint(2, 5)
        weights = None

        # Introduce class imbalance in ~30% of datasets
        if np.random.rand() < 0.3:
            major = np.random.uniform(0.7, 0.95)
            weights = [major] + [((1 - major) / (n_classes - 1))] * (n_classes - 1)

        try:
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_classes=n_classes,
                weights=weights,
                flip_y=0.01,
                class_sep=np.random.uniform(0.5, 2.0),
                random_state=random_seed + i,
            )
            X, y = shuffle(X, y, random_state=random_seed + i)

            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
            df["target"] = y

            filename = f"synthetic_dataset_{i+1}.csv"
            filepath = os.path.join(output_folder, filename)
            df.to_csv(filepath, index=False)

            print(f"✓ Saved: {filename} — {n_samples} samples, {n_features} features, {n_classes} classes")

        except Exception as e:
            print(f"✗ Failed to generate dataset {i+1}: {e}")

if __name__ == "__main__":
    generate_and_save_datasets()
