import numpy as np
import pandas as pd
import os
import glob
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy
import warnings
import joblib

warnings.filterwarnings('ignore')

class MetaFeatureExtractor:
    def __init__(self):
        self.meta_features = {}

    def extract_meta_features(self, X, y):
        meta_features = {}
        meta_features['n_samples'] = X.shape[0]
        meta_features['n_features'] = X.shape[1]
        meta_features['n_classes'] = len(np.unique(y))
        meta_features['samples_per_feature'] = X.shape[0] / X.shape[1]
        
        class_counts = np.bincount(y)
        meta_features['class_imbalance_ratio'] = np.max(class_counts) / np.min(class_counts)
        
        class_probs = class_counts / len(y)
        meta_features['dataset_entropy'] = entropy(class_probs)
        
        try:
            mi_scores = mutual_info_classif(X, y, random_state=42)
            meta_features['mutual_information_avg'] = np.mean(mi_scores)
            meta_features['mutual_information_max'] = np.max(mi_scores)
            meta_features['mutual_information_min'] = np.min(mi_scores)
        except:
            meta_features['mutual_information_avg'] = 0
            meta_features['mutual_information_max'] = 0
            meta_features['mutual_information_min'] = 0
        
        # Fixed signal-to-noise ratio calculation
        signal_to_noise = np.abs(np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        signal_to_noise = np.nan_to_num(signal_to_noise, nan=0.0, posinf=0.0, neginf=0.0)
        meta_features['signal_to_noise_ratio'] = np.mean(signal_to_noise)
        
        try:
            correlation_matrix = np.corrcoef(X.T)
            # Handle NaN values in correlation matrix
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            correlation_matrix = correlation_matrix[~np.eye(correlation_matrix.shape[0], dtype=bool)]
            meta_features['feature_correlation_avg'] = np.mean(np.abs(correlation_matrix))
            meta_features['feature_correlation_max'] = np.max(np.abs(correlation_matrix))
        except:
            meta_features['feature_correlation_avg'] = 0
            meta_features['feature_correlation_max'] = 0
        
        meta_features['fishers_discriminant_ratio'] = self._calculate_fishers_discriminant(X, y)
        
        # Clean all meta-features to ensure no infinite or NaN values
        for key, value in meta_features.items():
            if np.isnan(value) or np.isinf(value):
                meta_features[key] = 0.0
            meta_features[key] = float(meta_features[key])
        
        return meta_features

    def _calculate_fishers_discriminant(self, X, y):
        try:
            classes = np.unique(y)
            if len(classes) < 2:
                return 0.0
            
            overall_mean = np.mean(X, axis=0)
            within_class_scatter = 0
            between_class_scatter = 0
            
            for class_label in classes:
                class_data = X[y == class_label]
                if len(class_data) == 0:
                    continue
                    
                class_mean = np.mean(class_data, axis=0)
                class_size = len(class_data)
                
                within_class_scatter += np.sum((class_data - class_mean) ** 2)
                between_class_scatter += class_size * np.sum((class_mean - overall_mean) ** 2)
            
            # Avoid division by zero
            if within_class_scatter == 0 or within_class_scatter < 1e-10:
                return 1.0  # Return a reasonable default instead of infinity
            
            ratio = between_class_scatter / within_class_scatter
            
            # Cap the ratio to avoid extreme values
            ratio = min(ratio, 1000.0)
            
            return float(ratio)
        except:
            return 0.0

class DatasetLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.label_encoders = {}
    
    def load_datasets_from_folder(self):
        """Load all CSV datasets from the specified folder"""
        datasets = []
        
        # Get all CSV files
        csv_files = glob.glob(os.path.join(self.folder_path, "*.csv"))
        
        print(f"Found {len(csv_files)} CSV files")
        
        for file_path in csv_files:
            try:
                dataset_name = os.path.splitext(os.path.basename(file_path))[0]
                print(f"Loading dataset: {dataset_name}")
                
                X, y = self._load_csv_dataset(file_path, dataset_name)
                
                if X is not None and y is not None:
                    datasets.append((X, y, dataset_name))
                    print(f"  ✓ Successfully loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
                else:
                    print(f"  ✗ Failed to load: {dataset_name}")
                    
            except Exception as e:
                print(f"  ✗ Error loading {file_path}: {str(e)}")
        
        print(f"\nSuccessfully loaded {len(datasets)} out of {len(csv_files)} datasets")
        return datasets
    
    def _load_csv_dataset(self, file_path, dataset_name):
        """Load a single CSV dataset file"""
        try:
            # Try to read CSV with different configurations
            df = None
            
            # Common CSV configurations to try
            configs = [
                {'sep': ',', 'header': 0},      # Standard CSV with header
                {'sep': ',', 'header': None},   # CSV without header
                {'sep': ';', 'header': 0},      # European CSV format
                {'sep': ';', 'header': None},   # European CSV without header
                {'sep': '\t', 'header': 0},     # Tab-separated with header
                {'sep': '\t', 'header': None},  # Tab-separated without header
            ]
            
            for config in configs:
                try:
                    df = pd.read_csv(file_path, **config)
                    # Check if it looks reasonable
                    if df.shape[1] > 1 and df.shape[0] > 5:
                        break
                except:
                    continue
            
            if df is None:
                print(f"  Could not parse CSV file")
                return None, None
            
            # Remove any completely empty rows/columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df.empty:
                print(f"  Dataset is empty after cleaning")
                return None, None
            
            # Auto-detect target column
            target_col = self._identify_target_column(df)
            
            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Clean and preprocess features
            X = self._preprocess_features(X)
            
            # Clean and preprocess target
            y = self._preprocess_target(y, dataset_name)
            
            # Final validation
            if not self._validate_dataset(X, y):
                return None, None
            
            return X, y
            
        except Exception as e:
            print(f"  Error in _load_csv_dataset: {str(e)}")
            return None, None
    
    def _identify_target_column(self, df):
        """Identify which column is likely the target variable"""
        # Strategy 1: Look for common target column names
        target_names = ['class', 'target', 'label', 'y', 'output', 'category', 'result']
        for col in df.columns:
            if str(col).lower() in target_names:
                return col
        
        # Strategy 2: Find categorical columns with reasonable number of unique values
        categorical_candidates = []
        for col in df.columns:
            if df[col].dtype == 'object':
                categorical_candidates.append(col)
            elif df[col].nunique() <= 20 and df[col].nunique() >= 2:
                categorical_candidates.append(col)
        
        # Strategy 3: If last column is categorical, use it
        last_col = df.columns[-1]
        if last_col in categorical_candidates:
            return last_col
        
        # Strategy 4: Use first categorical candidate
        if categorical_candidates:
            return categorical_candidates[0]
        
        # Strategy 5: Default to last column
        return df.columns[-1]
    
    def _preprocess_features(self, X):
        """Preprocess feature columns"""
        # Handle case where X might already be a numpy array
        if isinstance(X, np.ndarray):
            X_array = X.astype(np.float32)
        else:
            # X is a pandas DataFrame
            for col in X.columns:
                if X[col].dtype == 'object':
                    # Try to convert to numeric first
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    except:
                        pass
                    
                    # If still object, encode categorically
                    if X[col].dtype == 'object':
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Convert to numpy array
            X_array = X.values.astype(np.float32)
        
        # Replace any remaining NaN values and infinities
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return X_array
    
    def _preprocess_target(self, y, dataset_name):
        """Preprocess target column"""
        # Handle non-numeric target
        if hasattr(y, 'dtype') and y.dtype == 'object':
            if dataset_name not in self.label_encoders:
                self.label_encoders[dataset_name] = LabelEncoder()
            y = self.label_encoders[dataset_name].fit_transform(y.astype(str))
        
        # Convert to numpy array - handle both pandas Series and numpy arrays
        if hasattr(y, 'values'):
            y_array = y.values.astype(np.int32)
        else:
            y_array = np.array(y).astype(np.int32)
        
        return y_array
    
    def _validate_dataset(self, X, y):
        """Validate that the dataset is suitable for classification"""
        # Check minimum number of samples
        if X.shape[0] < 10:
            print(f"  Dataset too small: {X.shape[0]} samples (minimum 10)")
            return False
        
        # Check number of features
        if X.shape[1] < 1:
            print(f"  No features found")
            return False
        
        # Check number of classes
        n_classes = len(np.unique(y))
        if n_classes < 2:
            print(f"  Not enough classes: {n_classes} (minimum 2)")
            return False
        
        # Check for valid data
        if np.any(np.isinf(X)) or np.any(np.isnan(X)):
            print(f"  Invalid values found in features")
            return False
        
        return True

class AlgorithmSelector:
    def __init__(self):
        self.algorithms = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'XGBoost': None,
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'KNN': KNeighborsClassifier(),
            'NaiveBayes': GaussianNB()
        }
        try:
            from xgboost import XGBClassifier
            self.algorithms['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss')
        except ImportError:
            print("XGBoost not available, using other algorithms")
            del self.algorithms['XGBoost']
        self.meta_feature_extractor = MetaFeatureExtractor()
        self.meta_classifier = None
        self.scaler = StandardScaler()
        self.meta_features_columns = None

    def evaluate_algorithm(self, X, y, algorithm, cv_folds=2):
        try:
            scores = cross_val_score(algorithm, X, y, cv=cv_folds, scoring='accuracy')
            return np.mean(scores)
        except:
            return 0.0

    def find_best_algorithm(self, X, y):
        scores = {}
        for name, algorithm in self.algorithms.items():
            if algorithm is not None:
                score = self.evaluate_algorithm(X, y, algorithm)
                scores[name] = score
        return max(scores, key=scores.get), scores

    def prepare_meta_dataset(self, datasets):
        meta_X, meta_y = [], []
        print("Preparing meta-dataset...")
        successful_datasets = 0
        
        for i, (X, y, name) in enumerate(datasets):
            try:
                print(f"Processing dataset {i+1}/{len(datasets)}: {name}")
                meta_features = self.meta_feature_extractor.extract_meta_features(X, y)
                best_algorithm, scores = self.find_best_algorithm(X, y)
                meta_X.append(list(meta_features.values()))
                meta_y.append(best_algorithm)
                successful_datasets += 1
                print(f"  Best algorithm: {best_algorithm}")
                print(f"  Scores: {scores}")
            except Exception as e:
                print(f"  Error processing {name}: {str(e)}")
                continue
        
        if successful_datasets == 0:
            raise ValueError("No datasets were successfully processed!")
        
        meta_X = np.array(meta_X)
        meta_y = np.array(meta_y)
        
        # Additional check for infinite or NaN values in meta_X
        meta_X = np.nan_to_num(meta_X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.meta_features_columns = list(meta_features.keys())
        
        print(f"Successfully processed {successful_datasets} datasets")
        print(f"Meta-dataset shape: {meta_X.shape}")
        print(f"Meta-dataset statistics:")
        print(f"  Min values: {np.min(meta_X, axis=0)}")
        print(f"  Max values: {np.max(meta_X, axis=0)}")
        print(f"  Any NaN: {np.any(np.isnan(meta_X))}")
        print(f"  Any Inf: {np.any(np.isinf(meta_X))}")
        
        return meta_X, meta_y

    def train_meta_classifier(self, meta_X, meta_y):
        print("Training meta-classifier...")
        
        # Double-check for problematic values before scaling
        print(f"Before scaling - NaN: {np.any(np.isnan(meta_X))}, Inf: {np.any(np.isinf(meta_X))}")
        
        # Clean the data one more time
        meta_X_clean = np.nan_to_num(meta_X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Fit and transform the scaler
        meta_X_scaled = self.scaler.fit_transform(meta_X_clean)
        
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        gb_classifier = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(
            gb_classifier,
            param_grid,
            cv=min(2, len(np.unique(meta_y))),  # Adjust CV folds based on available classes
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(meta_X_scaled, meta_y)
        self.meta_classifier = grid_search.best_estimator_
        # Save trained model and scaler
        joblib.dump(self.meta_classifier, 'meta_classifier.pkl')
        joblib.dump(self.scaler, 'meta_scaler.pkl')
        print("Saved meta-classifier to 'meta_classifier.pkl'")
        print("Saved scaler to 'meta_scaler.pkl'")

        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.meta_classifier

    def predict_best_algorithm(self, X, y):
        if self.meta_classifier is None:
            raise ValueError("Meta-classifier not trained yet!")
        
        meta_features = self.meta_feature_extractor.extract_meta_features(X, y)
        meta_features_array = np.array([list(meta_features.values())])
        
        # Clean the meta-features
        meta_features_array = np.nan_to_num(meta_features_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        meta_features_scaled = self.scaler.transform(meta_features_array)
        prediction = self.meta_classifier.predict(meta_features_scaled)[0]
        probabilities = self.meta_classifier.predict_proba(meta_features_scaled)[0]
        classes = self.meta_classifier.classes_
        
        sorted_indices = np.argsort(probabilities)[::-1]
        recommendations = []
        for idx in sorted_indices:
            recommendations.append({
                'algorithm': classes[idx],
                'probability': probabilities[idx]
            })
        
        return prediction, recommendations, meta_features

    def evaluate_meta_classifier(self, meta_X, meta_y, cv_folds=2):
        if self.meta_classifier is None:
            raise ValueError("Meta-classifier not trained yet!")
        
        # Clean the data
        meta_X_clean = np.nan_to_num(meta_X, nan=0.0, posinf=1e6, neginf=-1e6)
        meta_X_scaled = self.scaler.transform(meta_X_clean)
        
        cv_scores = cross_val_score(self.meta_classifier, meta_X_scaled, meta_y, cv=cv_folds)

        def top_k_accuracy(y_true, y_pred_proba, k=3):
            top_k_pred = np.argsort(y_pred_proba, axis=1)[:, -k:]
            correct = 0
            for i, true_label in enumerate(y_true):
                if true_label in self.meta_classifier.classes_[top_k_pred[i]]:
                    correct += 1
            return correct / len(y_true)

        y_pred_proba = self.meta_classifier.predict_proba(meta_X_scaled)
        accuracy = np.mean(cv_scores)
        top_3_accuracy = top_k_accuracy(meta_y, y_pred_proba, k=min(3, len(self.meta_classifier.classes_)))
        
        return {
            'accuracy': accuracy,
            'accuracy_std': np.std(cv_scores),
            'top_3_accuracy': top_3_accuracy,
            'cv_scores': cv_scores
        }

def main():
    print("=== Meta-Learning Based Algorithm Selection ===\n")
    
    # Specify the folder containing your datasets
    datasets_folder = input("Enter the path to your datasets folder: ").strip()
    
    if not os.path.exists(datasets_folder):
        print(f"Error: Folder '{datasets_folder}' does not exist!")
        return
    
    # Load datasets from folder
    loader = DatasetLoader(datasets_folder)
    datasets = loader.load_datasets_from_folder()
    
    if len(datasets) == 0:
        print("No datasets were successfully loaded!")
        return
    
    print(f"\nLoaded {len(datasets)} datasets for meta-learning")
    
    # Initialize algorithm selector
    selector = AlgorithmSelector()
    
    # Prepare meta-dataset
    meta_X, meta_y = selector.prepare_meta_dataset(datasets)
    print(f"\nMeta-dataset shape: {meta_X.shape}")
    print(f"Meta-features: {selector.meta_features_columns}")
    
    # Train meta-classifier
    selector.train_meta_classifier(meta_X, meta_y)
    
    # Evaluate meta-classifier
    evaluation_results = selector.evaluate_meta_classifier(meta_X, meta_y, cv_folds=2)
    print("\n=== Meta-Classifier Evaluation ===")
    print(f"Accuracy: {evaluation_results['accuracy']:.4f} ± {evaluation_results['accuracy_std']:.4f}")
    print(f"Top-3 Accuracy: {evaluation_results['top_3_accuracy']:.4f}")
    
    # Test on a new dataset (you can modify this part)
    print("\n=== Testing on New Dataset ===")
    
    # Option 1: Use one of the loaded datasets for testing
    if len(datasets) > 0:
        test_X, test_y, test_name = datasets[0]  # Use first dataset as example
        print(f"Testing on dataset: {test_name}")
        
        prediction, recommendations, meta_features = selector.predict_best_algorithm(test_X, test_y)
        print(f"Predicted best algorithm: {prediction}")
        print("\nTop recommendations:")
        for i, rec in enumerate(recommendations):
            print(f"  {i+1}. {rec['algorithm']}: {rec['probability']:.4f}")
        
        print(f"\nExtracted meta-features:")
        for feature, value in meta_features.items():
            print(f"  {feature}: {value:.4f}")
        
        # Verification
        print("\n=== Verification ===")
        actual_best, actual_scores = selector.find_best_algorithm(test_X, test_y)
        print(f"Actual best algorithm: {actual_best}")
        print(f"Actual scores: {actual_scores}")
        print(f"Prediction success: {prediction == actual_best}")

if __name__ == "__main__":
    main()