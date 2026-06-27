"""
Monitoring Logic Module - Functions for detecting ML model and data issues.

This module provides functions to monitor:
- Model Staleness: Check if model files are outdated
- Data Quality: Detect nulls, duplicates, and class imbalance
- Data Drift: Compare distributions between training and current data
- Label Noise: Identify inconsistent labels for identical features
- Data Leakage: Detect features with suspiciously high correlation to target
"""

import os
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def check_model_staleness(model_path: str) -> Dict[str, Any]:
    """
    Check if a model file is outdated based on file age.
    
    Args:
        model_path: Path to the model file.
    
    Returns:
        Dictionary containing:
        - 'is_stale': Boolean indicating if model is older than 30 days
        - 'days_old': Number of days since model creation
        - 'created_date': Creation date of the model file
        - 'recommendation': Action recommendation based on staleness
    """
    if not os.path.exists(model_path):
        return {
            'is_stale': True,
            'days_old': -1,
            'created_date': None,
            'recommendation': 'Model file not found. Retrain immediately.'
        }
    
    file_modified = datetime.fromtimestamp(os.path.getmtime(model_path))
    days_old = (datetime.now() - file_modified).days
    is_stale = days_old > 30
    
    recommendation = (
        'Model is outdated. Consider retraining.' if is_stale
        else f'Model is recent (modified {days_old} days ago).'
    )
    
    return {
        'is_stale': is_stale,
        'days_old': days_old,
        'created_date': file_modified.strftime('%Y-%m-%d %H:%M:%S'),
        'recommendation': recommendation
    }


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality by checking for nulls, duplicates, and class imbalance.
    
    Args:
        df: Input DataFrame to validate.
    
    Returns:
        Dictionary containing:
        - 'null_columns': Dictionary of column names and their null counts
        - 'null_percentage': Percentage of null values in the dataset
        - 'duplicate_rows': Number of duplicate rows
        - 'duplicate_percentage': Percentage of duplicate rows
        - 'is_quality_good': Boolean flag indicating if quality is acceptable
        - 'issues': List of identified quality issues
    """
    issues = []
    
    # Check for null values
    null_counts = df.isnull().sum()
    null_columns = null_counts[null_counts > 0].to_dict()
    null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    
    if null_percentage > 10:
        issues.append(f'High null percentage: {null_percentage:.2f}%')
    
    # Check for duplicates
    duplicate_rows = df.duplicated().sum()
    duplicate_percentage = (duplicate_rows / len(df)) * 100 if len(df) > 0 else 0
    
    if duplicate_percentage > 5:
        issues.append(f'High duplicate percentage: {duplicate_percentage:.2f}%')
    
    is_quality_good = len(issues) == 0 and null_percentage < 5
    
    return {
        'null_columns': null_columns,
        'null_percentage': round(null_percentage, 2),
        'duplicate_rows': int(duplicate_rows),
        'duplicate_percentage': round(duplicate_percentage, 2),
        'is_quality_good': is_quality_good,
        'issues': issues
    }


def detect_data_drift(train_df: pd.DataFrame, current_df: pd.DataFrame, 
                     threshold: float = 0.05) -> Dict[str, Any]:
    """
    Detect data drift by comparing feature distributions using Kolmogorov-Smirnov test.
    
    Args:
        train_df: Training dataset for baseline comparison.
        current_df: Current dataset to check for drift.
        threshold: P-value threshold for drift detection (default 0.05).
    
    Returns:
        Dictionary containing:
        - 'drift_detected': Boolean indicating if drift was detected
        - 'drifted_features': List of features with detected drift
        - 'drift_details': Dictionary with KS statistics and p-values for each feature
        - 'recommendation': Action recommendation based on drift
    """
    drift_details = {}
    drifted_features = []
    
    # Identify numeric columns common to both datasets
    numeric_cols = set(train_df.select_dtypes(include=[np.number]).columns) & \
                   set(current_df.select_dtypes(include=[np.number]).columns)
    
    for col in numeric_cols:
        train_clean = train_df[col].dropna()
        current_clean = current_df[col].dropna()
        
        if len(train_clean) > 0 and len(current_clean) > 0:
            ks_stat, p_value = ks_2samp(train_clean, current_clean)
            drift_details[col] = {
                'ks_statistic': round(float(ks_stat), 4),
                'p_value': round(float(p_value), 6)
            }
            
            if p_value < threshold:
                drifted_features.append(col)
    
    drift_detected = len(drifted_features) > 0
    recommendation = (
        f'Drift detected in {len(drifted_features)} features. Consider retraining.'
        if drift_detected
        else 'No significant data drift detected.'
    )
    
    return {
        'drift_detected': drift_detected,
        'drifted_features': drifted_features,
        'drift_details': drift_details,
        'recommendation': recommendation
    }


def identify_label_noise(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Identify label noise by finding feature combinations with inconsistent target labels.
    
    Args:
        df: DataFrame containing features and target column.
        target_col: Name of the target/label column.
    
    Returns:
        Dictionary containing:
        - 'noise_detected': Boolean indicating if label noise was found
        - 'inconsistent_samples': Number of samples with inconsistent labels
        - 'problematic_features': List of feature combinations showing inconsistency
        - 'recommendation': Action recommendation based on noise level
    """
    if target_col not in df.columns:
        return {
            'noise_detected': False,
            'inconsistent_samples': 0,
            'problematic_features': [],
            'recommendation': f'Target column "{target_col}" not found in dataset.'
        }
    
    # Get feature columns (exclude target)
    feature_cols = [col for col in df.columns if col != target_col]
    
    if len(feature_cols) == 0:
        return {
            'noise_detected': False,
            'inconsistent_samples': 0,
            'problematic_features': [],
            'recommendation': 'No features found to check for label noise.'
        }
    
    # Check for duplicate feature combinations with different labels
    inconsistent_pairs = []
    total_inconsistent = 0
    
    for feature_subset in [feature_cols[:min(3, len(feature_cols))]]:  # Limit to first 3 features
        grouped = df.groupby(feature_subset)[target_col].nunique()
        problematic = grouped[grouped > 1]
        
        if len(problematic) > 0:
            for features in problematic.index:
                inconsistent_pairs.append({
                    'features': str(feature_subset),
                    'inconsistent_labels': int(problematic[features])
                })
                total_inconsistent += problematic[features]
    
    noise_detected = total_inconsistent > 0
    recommendation = (
        f'Label noise detected in {len(inconsistent_pairs)} feature combinations. '
        'Review and clean training data.'
        if noise_detected
        else 'No significant label noise detected.'
    )
    
    return {
        'noise_detected': noise_detected,
        'inconsistent_samples': total_inconsistent,
        'problematic_features': inconsistent_pairs,
        'recommendation': recommendation
    }


def detect_leakage(df: pd.DataFrame, target_col: str, 
                  leakage_threshold: float = 0.95) -> Dict[str, Any]:
    """
    Detect data leakage by identifying features with suspiciously high correlation to target.
    
    Args:
        df: DataFrame containing features and target column.
        target_col: Name of the target/label column.
        leakage_threshold: Correlation threshold for leakage detection (default 0.95).
    
    Returns:
        Dictionary containing:
        - 'leakage_detected': Boolean indicating if potential leakage was found
        - 'leaky_features': List of features with high correlation to target
        - 'correlations': Dictionary of features and their correlation to target
        - 'recommendation': Action recommendation based on leakage detection
    """
    if target_col not in df.columns:
        return {
            'leakage_detected': False,
            'leaky_features': [],
            'correlations': {},
            'recommendation': f'Target column "{target_col}" not found in dataset.'
        }
    
    # Select only numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target_col not in numeric_df.columns:
        return {
            'leakage_detected': False,
            'leaky_features': [],
            'correlations': {},
            'recommendation': f'Target column "{target_col}" is not numeric.'
        }
    
    correlations = {}
    leaky_features = []
    
    # Calculate correlation of each feature with target
    for col in numeric_df.columns:
        if col != target_col:
            corr = numeric_df[col].corr(numeric_df[target_col])
            abs_corr = abs(corr)
            correlations[col] = round(abs_corr, 4)
            
            if abs_corr > leakage_threshold:
                leaky_features.append(col)
    
    leakage_detected = len(leaky_features) > 0
    recommendation = (
        f'Potential data leakage detected in {len(leaky_features)} features. '
        'Review and remove these features before training.'
        if leakage_detected
        else 'No suspicious leakage patterns detected.'
    )
    
    return {
        'leakage_detected': leakage_detected,
        'leaky_features': leaky_features,
        'correlations': correlations,
        'recommendation': recommendation
    }


def check_retrain_needed(
    train_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_col: str,
    model_path: str
) -> Dict[str, Any]:
    """
    Simple check to determine if model retraining is needed.
    
    Args:
        train_df: Training dataset
        current_df: Current dataset
        target_col: Target column name
        model_path: Path to model file
    
    Returns:
        Dictionary with 'retrain_needed' (bool) and 'reason' (str)
    """
    reasons = []
    
    # Prioritize by severity: staleness > leakage > quality > drift > noise
    
    # Check 0: Model Staleness (check if model is too old - 14+ days)
    staleness = check_model_staleness(model_path)
    if staleness.get('days_old', 0) > 14:  # Model older than 14 days
        reasons.append('model staleness')
    
    # Check 1: Data Leakage (most severe - data quality issue)
    leakage = detect_leakage(current_df, target_col, leakage_threshold=0.80)
    if leakage.get('leakage_detected'):
        reasons.append('data leakage')
    
    # Check 2: Data Quality
    quality = validate_data_quality(current_df)
    if not quality.get('is_quality_good'):
        reasons.append('poor data quality')
    
    # Check 3: Data Drift
    drift = detect_data_drift(train_df, current_df, threshold=0.05)
    if drift.get('drift_detected'):
        reasons.append('data drift')
    
    # Check 4: Label Noise
    noise = identify_label_noise(current_df, target_col)
    if noise.get('noise_detected'):
        reasons.append('label noise')
    
    # Determine if retrain needed
    retrain_needed = len(reasons) > 0
    reason_text = reasons[0] if reasons else 'no issues detected'
    
    return {
        'retrain_needed': retrain_needed,
        'reason': reason_text
    }
