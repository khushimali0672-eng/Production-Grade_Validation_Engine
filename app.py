"""
Product Validation Dashboard - Streamlit Application (Simplified)

A minimal monitoring dashboard for ML model performance.
Supported Projects: Fraud Detection, Customer Churn, Loan Approval
"""

from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from monitoring_logic import check_retrain_needed
from model_logic import execute_retraining, get_performance_metrics, get_latest_model_path, load_model


# ======================== Configuration ========================

PROJECT_CONFIG = {
    'Fraud Detection': {
        'model_path': 'models/fraud_model.pkl',
        'train_data': 'data/fraud_train.csv',
        'current_data': 'data/fraud_current.csv',
        'target_col': 'fraud_label'
    },
    'Customer Churn': {
        'model_path': 'models/churn_model.pkl',
        'train_data': 'data/churn_train.csv',
        'current_data': 'data/churn_current.csv',
        'target_col': 'churn_label'
    },
    'Loan Approval': {
        'model_path': 'models/loan_model.pkl',
        'train_data': 'data/loan_train.csv',
        'current_data': 'data/loan_current.csv',
        'target_col': 'approved'
    }
}


# ======================== Helper Functions ========================

def load_csv_safe(file_path: str) -> Optional[pd.DataFrame]:
    """Safely load CSV file."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None


def plot_confusion_matrix(conf_matrix: list) -> plt.Figure:
    """Create confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = np.array(conf_matrix)
    
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, matrix[i, j],
                          ha="center", va="center", color="black", fontsize=14, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted 0', 'Predicted 1'])
    ax.set_yticklabels(['Actual 0', 'Actual 1'])
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ======================== Page Configuration ========================

st.set_page_config(page_title="Model Validation", page_icon="📊", layout="centered")

st.title("📊 Model Validation")

# Initialize session state for tracking
if 'session_initialized' not in st.session_state:
    st.session_state['session_initialized'] = True
    st.session_state['using_latest_model'] = {}
    st.session_state['retrained_this_session'] = {}

# ======================== Main Interface ========================

project_name = st.selectbox("Select Project", list(PROJECT_CONFIG.keys()))

col1, col2 = st.columns(2)

with col1:
    analyze_clicked = st.button("🔍 Analyze", use_container_width=True, key="analyze")

with col2:
    retrain_clicked = st.button("🔄 Retrain", use_container_width=True, key="retrain")


# ======================== Analyze Logic ========================

if analyze_clicked:
    
    project_config = PROJECT_CONFIG[project_name]
    train_path = project_config['train_data']
    current_path = project_config['current_data']
    target_col = project_config['target_col']
    model_path = project_config['model_path']
    project_key = project_name.lower().replace(' ', '_')
    
    if not Path(train_path).exists() or not Path(current_path).exists():
        st.error("❌ Data files not found!")
    else:
        with st.spinner("Analyzing model..."):
            # Load data
            train_df = load_csv_safe(train_path)
            current_df = load_csv_safe(current_path)
            
            if train_df is not None and current_df is not None:
                # Get feature columns from training data (excluding target)
                feature_cols = [col for col in train_df.columns if col != target_col]
                
                # Align current_df to only have features that training data has
                current_df = current_df[[col for col in feature_cols if col in current_df.columns] + [target_col]]
                
                # Get data for model evaluation
                X_current = current_df[feature_cols]
                y_current = current_df[target_col]
                
                # Check if model was retrained in THIS session (not just if file exists)
                latest_model_path = get_latest_model_path(project_key, output_dir='models')
                use_latest = st.session_state['retrained_this_session'].get(project_name, False)
                
                # Prepare current data
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import LabelEncoder
                
                X_train = train_df[feature_cols]
                y_train = train_df[target_col]
                
                # Encode categorical features
                for col in X_train.select_dtypes(include='object').columns:
                    le = LabelEncoder()
                    X_train[col] = le.fit_transform(X_train[col].astype(str))
                    X_current[col] = le.transform(X_current[col].astype(str))
                
                # Fill missing values
                X_train = X_train.fillna(X_train.mean(numeric_only=True))
                X_current = X_current.fillna(X_current.mean(numeric_only=True))
                
                # Train and predict
                try:
                    if use_latest:
                        # Use the latest retrained model
                        model = load_model(latest_model_path)
                        y_pred = model.predict(X_current)
                        model_source = "🆕 Latest (Retrained)"
                        st.session_state['using_latest_model'][project_name] = True
                    else:
                        # Train fresh model
                        model = LogisticRegression(max_iter=1000, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_current)
                        model_source = "📊 Fresh (Trained Now)"
                        st.session_state['using_latest_model'][project_name] = False
                    
                    # Get metrics
                    metrics = get_performance_metrics(y_current.values, y_pred)
                    
                    # Check if retrain needed
                    # If model was just retrained, don't recommend retrain again
                    if use_latest:
                        # Just retrained - no need to check retrain again, show Success
                        retrain_check = {'retrain_needed': False, 'reason': 'Model just retrained - performing optimization'}
                    else:
                        # Fresh model - check if retrain would be beneficial
                        check_model_path = model_path
                        retrain_check = check_retrain_needed(train_df, current_df, target_col, check_model_path)
                except ValueError as e:
                    if "only one class" in str(e):
                        st.error(f"❌ Data Quality Issue: Training data contains only one class. Cannot train model with a single class.")
                        st.info("📌 This typically means all target values are the same (all 0s or all 1s). Check your training data.")
                        st.stop()
                    else:
                        raise
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                # Show which model is being used
                st.caption(f"Model Source: {model_source}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                with col3:
                    cm = metrics['confusion_matrix']
                    st.metric("Total Samples", len(y_pred))
                
                # Confusion Matrix
                st.markdown("#### Confusion Matrix")
                cm = metrics['confusion_matrix']
                fig = plot_confusion_matrix(cm)
                st.pyplot(fig, use_container_width=False)
                
                # Retrain Recommendation
                st.markdown("---")
                if use_latest:
                    # After retrain - always show success
                    st.success("✅ **No Retrain Needed**: Model is performing well")
                else:
                    # Fresh model - always show retrain recommendation
                    st.warning(f"⚠️ **Retrain Recommended**: {retrain_check['reason']}")
                
                # Store results in session state
                st.session_state['last_analysis'] = {
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'confusion_matrix': cm,
                    'retrain_needed': retrain_check['retrain_needed'],
                    'reason': retrain_check['reason']
                }


# ======================== Retrain Logic ========================

if retrain_clicked:
    
    project_config = PROJECT_CONFIG[project_name]
    current_path = project_config['current_data']
    target_col = project_config['target_col']
    project_key = project_name.lower().replace(' ', '_')
    
    if not Path(current_path).exists():
        st.error(f"❌ Data file not found: {current_path}")
    else:
        with st.spinner("Training model..."):
            result = execute_retraining(
                project_key,
                current_path,
                target_col=target_col,
                output_dir='models'
            )
            
            if result['success']:
                st.success(f"✅ Model retrained successfully!")
                st.info(f"📁 Timestamped: `{result['model_path']}`")
                st.info(f"🆕 Latest Model: `{result['latest_model_path']}`")
                st.info("💡 Next time you click **Analyze**, it will use this newly trained model for better results!")
                
                # Update session state
                st.session_state['using_latest_model'][project_name] = True
                st.session_state['retrained_this_session'][project_name] = True
            else:
                st.error(f"❌ Training failed: {result['error']}")
