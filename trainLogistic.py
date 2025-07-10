import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

# === Load meta-dataset ===
df = pd.read_csv("meta_dataset.csv")

# Separate features and labels
X = df.drop(columns=["best_algorithm"]).values
y = df["best_algorithm"].values

# Clean feature matrix
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train classifier ===
clf = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    multi_class='multinomial',
    solver='lbfgs',
    random_state=42
)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    clf,
    param_grid,
    cv=min(2, len(np.unique(y))),
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

grid.fit(X_scaled, y)

# === Save model and scaler ===
joblib.dump(grid.best_estimator_, "meta_classifier.pkl")
joblib.dump(scaler, "meta_scaler.pkl")

print("✅ Trained and saved model.")
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
