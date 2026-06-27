"""
Model Operations Logic Module - Functions for model training and performance evaluation.

This module provides functions for:
- Calculating performance metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
- Executing model retraining pipeline with timestamped model saving
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def get_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics for binary classification.
    
    Args:
        y_true: Ground truth labels (binary: 0 or 1).
        y_pred: Predicted labels from model.
    
    Returns:
        Dictionary containing:
        - 'accuracy': Overall accuracy of predictions
        - 'precision': Precision score (TP / TP+FP)
        - 'recall': Recall/Sensitivity (TP / TP+FN)
        - 'f1_score': F1 score (harmonic mean of precision and recall)
        - 'confusion_matrix': 2x2 confusion matrix
        - 'classification_report': Detailed classification metrics by class
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'confusion_matrix': [[0, 0], [0, 0]],
            'classification_report': 'No data available'
        }
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0, average='binary')
    recall = recall_score(y_true, y_pred, zero_division=0, average='binary')
    f1 = f1_score(y_true, y_pred, zero_division=0, average='binary')
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=False)
    
    return {
        'accuracy': round(float(accuracy), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1_score': round(float(f1), 4),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    }


def _prepare_data_for_training(
    csv_path: str,
    target_col: str,
    test_size: float = 0.2,
    reference_features: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Internal function to load and prepare data for model training.
    
    Args:
        csv_path: Path to the CSV file containing training data.
        target_col: Name of the target/label column.
        test_size: Fraction of data to use for testing (default 0.2).
        reference_features: Optional list of features to use. If provided, only these features are kept.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names).
    
    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If target column is not found in the dataset.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Filter to reference features if provided
    if reference_features:
        available_features = [col for col in reference_features if col in X.columns]
        X = X[available_features]
    
    # Handle non-numeric features
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_processed = X.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    
    # Handle missing values
    X_processed = X_processed.fillna(X_processed.mean(numeric_only=True))
    
    feature_names = list(X_processed.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=42, stratify=y
    )
    
    return X_train.values, X_test.values, y_train.values, y_test.values, feature_names


def _train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000
) -> LogisticRegression:
    """
    Internal function to train a Logistic Regression model.
    
    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        max_iter: Maximum iterations for solver (default 1000).
    
    Returns:
        Trained LogisticRegression model.
    """
    model = LogisticRegression(max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    return model


def execute_retraining(
    project_name: str,
    csv_path: str,
    target_col: str = 'target',
    output_dir: str = 'models'
) -> Dict[str, Any]:
    """
    Execute the complete retraining pipeline for a project.

    Trains a new Logistic Regression model on the provided data,
    evaluates it on a test set, and saves the model with a timestamp.
    
    Args:
        project_name: Name of the project (e.g., 'fraud_detection').
        csv_path: Path to the CSV file containing training data (current data).
        target_col: Name of the target/label column (default 'target').
        output_dir: Directory to save the retrained model (default 'models').
    
    Returns:
        Dictionary containing:
        - 'success': Boolean indicating if retraining was successful
        - 'model_path': Path where the model was saved
        - 'timestamp': Timestamp when the model was created
        - 'train_metrics': Performance metrics on training data
        - 'test_metrics': Performance metrics on test data
        - 'error': Error message if retraining failed (None if successful)
    
    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If target column is not found in the dataset.
    """
    try:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get reference features from the original training data
        # Replace 'current' with 'train' to find the training data path
        train_csv_path = csv_path.replace('_current.csv', '_train.csv')
        
        # Load training data to get feature list
        train_df = pd.read_csv(train_csv_path)
        train_features = [col for col in train_df.columns if col != target_col]
        
        # Prepare retraining data using the same features as training data
        X_train, X_test, y_train, y_test, feature_names = _prepare_data_for_training(
            csv_path, target_col, reference_features=train_features
        )
        
        # Train model
        model = _train_logistic_regression(X_train, y_train)
        
        # Evaluate on training and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_metrics = get_performance_metrics(y_train, y_train_pred)
        test_metrics = get_performance_metrics(y_test, y_test_pred)
        
        # Save model with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{project_name}_model_{timestamp}.pkl"
        model_path = str(Path(output_dir) / model_filename)
        
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Also save as latest model for inference
        latest_model_name = f"{project_name}_model_latest.pkl"
        latest_model_path = str(Path(output_dir) / latest_model_name)
        with open(latest_model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Clean up old models, keeping only the last 5
        _cleanup_old_models(project_name, output_dir, keep_count=5)
        
        return {
            'success': True,
            'model_path': model_path,
            'latest_model_path': latest_model_path,
            'timestamp': timestamp,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'error': None
        }
    
    except FileNotFoundError as e:
        return {
            'success': False,
            'model_path': None,
            'latest_model_path': None,
            'timestamp': None,
            'train_metrics': None,
            'test_metrics': None,
            'error': str(e)
        }
    
    except ValueError as e:
        return {
            'success': False,
            'model_path': None,
            'latest_model_path': None,
            'timestamp': None,
            'train_metrics': None,
            'test_metrics': None,
            'error': str(e)
        }
    
    except Exception as e:
        return {
            'success': False,
            'model_path': None,
            'latest_model_path': None,
            'timestamp': None,
            'train_metrics': None,
            'test_metrics': None,
            'error': f'Unexpected error during retraining: {str(e)}'
        }


def _cleanup_old_models(
    project_name: str,
    output_dir: str = 'models',
    keep_count: int = 5
) -> None:
    """
    Remove old model files, keeping only the last N versions.
    
    Args:
        project_name: Name of the project (e.g., 'fraud_detection').
        output_dir: Directory where models are stored (default 'models').
        keep_count: Number of recent models to keep (default 5).
    """
    try:
        models_dir = Path(output_dir)
        if not models_dir.exists():
            return
        
        # Find all timestamped models for this project (exclude _latest models)
        pattern = f"{project_name}_model_*.pkl"
        model_files = [
            f for f in models_dir.glob(pattern)
            if not str(f.name).endswith('_latest.pkl')
        ]
        
        if len(model_files) <= keep_count:
            return
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Delete old models beyond keep_count
        for old_model in model_files[keep_count:]:
            try:
                old_model.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {old_model}: {e}")
    
    except Exception as e:
        print(f"Warning: Error during model cleanup: {e}")


def get_latest_model_path(project_name: str, output_dir: str = 'models') -> Optional[str]:
    """
    Get path to the latest trained model for a project.
    
    Args:
        project_name: Name of the project (e.g., 'fraud_detection').
        output_dir: Directory where models are stored (default 'models').
    
    Returns:
        Path to latest model if it exists, None otherwise.
    """
    latest_model_name = f"{project_name}_model_latest.pkl"
    latest_model_path = Path(output_dir) / latest_model_name
    return str(latest_model_path) if latest_model_path.exists() else None


def load_model(model_path: str) -> Optional[Any]:
    """
    Load a previously saved model from disk.
    
    Args:
        model_path: Path to the saved model file.
    
    Returns:
        The loaded model object, or None if loading fails.
    """
    try:
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
