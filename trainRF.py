import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

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

# === Train Random Forest classifier ===
clf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'  # handle class imbalance
)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid = GridSearchCV(
    clf,
    param_grid,
    cv=min(3, len(np.unique(y))),  # Use 3-fold CV or less if classes < 3
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

grid.fit(X_scaled, y)

# === Save model and scaler ===
joblib.dump(grid.best_estimator_, "meta_classifier.pkl")
joblib.dump(scaler, "meta_scaler.pkl")

print("✅ Trained and saved Random Forest model.")
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
