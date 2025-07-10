import numpy as np
import joblib
import os
from meta_learning_custom_3 import DatasetLoader, MetaFeatureExtractor, AlgorithmSelector

def evaluate_saved_meta_model(datasets_folder):
    # Load trained model and scaler
    model = joblib.load("meta_classifier.pkl")
    scaler = joblib.load("meta_scaler.pkl")

    # Initialize components
    loader = DatasetLoader(datasets_folder)
    extractor = MetaFeatureExtractor()
    selector = AlgorithmSelector()  # For evaluating actual best algorithm

    datasets = loader.load_datasets_from_folder()
    if not datasets:
        print("No datasets found or loaded.")
        return

    correct_top1 = 0
    correct_top3 = 0
    total = 0
    prediction_distribution = {}

    print("\n=== Meta-Model Evaluation ===")

    for X, y, name in datasets:
        try:
            # Extract meta-features
            meta_features = extractor.extract_meta_features(X, y)
            meta_array = np.array([list(meta_features.values())])
            meta_array = np.nan_to_num(meta_array, nan=0.0, posinf=1e6, neginf=-1e6)
            meta_scaled = scaler.transform(meta_array)

            # Meta-model prediction
            predicted = model.predict(meta_scaled)[0]
            probabilities = model.predict_proba(meta_scaled)[0]
            classes = model.classes_

            # Top-3 predictions
            top3_indices = np.argsort(probabilities)[::-1][:3]
            top3_preds = [classes[i] for i in top3_indices]

            # Actual best algorithm
            actual, _ = selector.find_best_algorithm(X, y)

            print(f"{name}: Predicted = {predicted}, Actual = {actual}")
            print("  Top-3:", ", ".join(f"{alg} ({probabilities[i]:.2f})" for i, alg in zip(top3_indices, top3_preds)))

            if predicted == actual:
                correct_top1 += 1
            if actual in top3_preds:
                correct_top3 += 1

            prediction_distribution[predicted] = prediction_distribution.get(predicted, 0) + 1
            total += 1

        except Exception as e:
            print(f"Error processing {name}: {e}")

    # Final summary
    print(f"\nTop-1 Accuracy: {correct_top1}/{total} = {correct_top1 / total:.4f}")
    print(f"Top-3 Accuracy: {correct_top3}/{total} = {correct_top3 / total:.4f}")

    print("\nMeta-Model Prediction Distribution:")
    for alg, count in prediction_distribution.items():
        print(f"  {alg}: {count} times")

if __name__ == "__main__":
    folder = input("Enter path to dataset folder: ").strip()
    if not os.path.isdir(folder):
        print("Invalid folder path.")
    else:
        evaluate_saved_meta_model(folder)
