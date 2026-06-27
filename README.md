# Product Validation Framework

A lightweight ML model monitoring framework with automated retraining pipeline. Detects data issues, model staleness, and provides an interactive Streamlit dashboard.

**Key Features:**
- 🔍 **Monitoring**: Model staleness, data quality, drift, leakage, label noise
- 📊 **Dashboard**: Interactive Streamlit UI with visualization
- 🔄 **Auto-Retrain**: Automated pipeline with timestamped model saving
- 💻 **Pure Functions**: No classes - clean, testable, functional approach

**Supported Projects:**
- 🚨 Fraud Detection | 👥 Customer Churn | 💰 Loan Approval

---

## ⚡ Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

Opens at: `http://localhost:8501`

**Usage:**
1. Select project from dropdown
2. Click **🔍 Analyze** → view metrics & recommendations
3. Click **🔄 Retrain** → train new model (auto-saved with timestamp)

---

## 📁 Project Structure

```
prod_validation/
├── app.py                      # Streamlit dashboard
├── monitoring_logic.py         # Monitoring functions
├── model_logic.py              # Model training functions
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── data/                       # CSV files (6 files)
├── models/                     # Saved models
└── .venv/                      # Virtual environment
```

---

## 📊 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place CSV files in `data/` directory:
- `fraud_train.csv` / `fraud_current.csv`
- `churn_train.csv` / `churn_current.csv`
- `loan_train.csv` / `loan_current.csv`

**CSV Format:**
```
feature_1,feature_2,feature_3,...,target_column
1.5,0,23,...,0
2.3,1,45,...,1
```

- Target columns: `fraud_label`, `churn_label`, `approved`
- Must have numeric or categorical features + binary target (0/1)

---

## 🔧 Core Modules

### `monitoring_logic.py` (6 functions)

| Function | Purpose |
|----------|---------|
| `check_model_staleness()` | Detects if model > 30 days old |
| `validate_data_quality()` | Checks nulls, duplicates, imbalance |
| `detect_data_drift()` | Compares distributions (Kolmogorov-Smirnov) |
| `identify_label_noise()` | Finds inconsistent labels |
| `detect_leakage()` | Flags features with correlation > 0.80 |
| `check_retrain_needed()` | Prioritizes issues (staleness > leakage > quality > drift > noise) |

### `model_logic.py` (6 functions)

| Function | Purpose |
|----------|---------|
| `get_performance_metrics()` | Returns: Accuracy, Precision, Recall, F1, Confusion Matrix |
| `execute_retraining()` | Trains model, saves with timestamp |
| `load_model()` | Loads pickled model |
| `get_latest_model_path()` | Finds latest model for project |
| `_prepare_data_for_training()` | Internal preprocessing |
| `_train_logistic_regression()` | Internal training |

### `app.py`

**Dashboard Controls:**
- Project selector (dropdown)
- **🔍 Analyze** - Runs monitoring pipeline
- **🔄 Retrain** - Trains new model

**Results Display:**
- Accuracy, Precision, Confusion Matrix
- Retrain recommendation (one-line reason)

---

## 📈 Monitoring Checks Explained

### Model Staleness
- Checks file modification date
- ⚠️ Alerts if > 30 days old

### Data Quality
- Nulls: flags if > 10%
- Duplicates: flags if > 5%
- Class imbalance: flags major skew

### Data Drift
- Uses Kolmogorov-Smirnov (KS) test
- Threshold: p-value < 0.05
- Compares numeric feature distributions

### Label Noise
- Finds identical features with different labels
- Indicates mislabeling or inconsistencies

### Data Leakage
- Calculates feature-target correlations
- ⚠️ Flags if correlation > 0.80
- Indicates unavailable-at-prediction-time information

### Retrain Priority
```
Severity: Staleness > Leakage > Quality > Drift > Noise
Returns: highest-priority issue
```

---

## 🎯 Usage Examples

### Example 1: Dashboard Usage
```bash
streamlit run app.py
# 1. Select project
# 2. Click Analyze
# 3. Review results & recommendation
# 4. Click Retrain (if recommended)
```

### Example 2: Python API
```python
from monitoring_logic import check_retrain_needed
from model_logic import execute_retraining
import pandas as pd

# Check if retrain needed
train_df = pd.read_csv('data/fraud_train.csv')
current_df = pd.read_csv('data/fraud_current.csv')
result = check_retrain_needed(train_df, current_df, 'fraud_label', 'models/model.pkl')

if result['retrain_needed']:
    print(f"Issue: {result['reason']}")
    
    # Retrain
    retrain = execute_retraining('fraud', 'data/fraud_current.csv', 'fraud_label')
    print(f"New model: {retrain['model_path']}")
    print(f"Test accuracy: {retrain['test_metrics']['accuracy']}")
```

### Example 3: Individual Monitoring Functions
```python
from monitoring_logic import detect_leakage, detect_data_drift
import pandas as pd

df = pd.read_csv('data/fraud_current.csv')
train_df = pd.read_csv('data/fraud_train.csv')

# Check leakage
leakage = detect_leakage(df, 'fraud_label', leakage_threshold=0.80)
if leakage['leakage_detected']:
    print(f"⚠️ Leaky features: {leakage['leaky_features']}")

# Check drift
drift = detect_data_drift(train_df, df, threshold=0.05)
if drift['drift_detected']:
    print(f"⚠️ Drifted features: {drift['drifted_features']}")
```

---

## ⚙️ Configuration

Edit **PROJECT_CONFIG** in `app.py`:

```python
PROJECT_CONFIG = {
    'Your Project': {
        'model_path': 'models/your_model.pkl',
        'train_data': 'data/your_train.csv',
        'current_data': 'data/your_current.csv',
        'target_col': 'your_target_column'
    }
}
```

**Thresholds (in monitoring functions):**
- **Drift threshold**: p-value < 0.05 (default) - lower = more sensitive
- **Leakage threshold**: correlation > 0.80 (default) - lower = more sensitive
- **Model staleness**: days > 30 (default)
- **Quality issues**: nulls > 10%, duplicates > 5%

---

## 📊 Data & Accuracy Analysis

### Why Realistic Accuracy (Not 100%)

Real ML systems should NOT have 100% accuracy:
- ❌ **100% = Red flag** → indicates leakage, overfitting, or test-on-train
- ✅ **80-85% = Realistic** → real-world data has noise, missing factors
- ✅ **65-75% without leakage** → honest assessment of feature quality

**Our Loan Approval Example:**
```
With leakage (decision_confidence at 0.84 correlation): 80-85% accuracy
Without leakage (base features only): 65-75% accuracy
→ Demonstrates why leakage detection is critical
```

### Understanding Feature Correlations

- `credit_score`: 0.22 correlation (weak but useful)
- `decision_confidence` (leakage): 0.84 correlation (too perfect - red flag!)
- Other features: < 0.06 correlation

**Leakage signals:**
- Features with extremely high correlation (> 0.80)
- Features appearing in training but not in production
- Information only available after decision

### Retrain Reasons

| Reason | Severity | Action |
|--------|----------|--------|
| Model Staleness | High | Retrain regularly (30+ days) |
| **Data Leakage** | **Critical** | **Remove leaky features immediately** |
| Poor Data Quality | High | Fix data issues first |
| Data Drift | Medium | Retrain with new distribution |
| Label Noise | Low | Review & clean labels |

---

## 🐛 Troubleshooting

**"CSV file not found"**
```
✓ Check files exist in data/ directory
✓ Verify filenames in PROJECT_CONFIG
✓ Check working directory
```

**"Target column not found"**
```
✓ Verify column name matches CSV header (case-sensitive)
✓ Check target_col in PROJECT_CONFIG
```

**"Model training fails"**
```
✓ Ensure CSV has both features and target
✓ Check target is binary (0/1)
✓ Verify no all-NaN columns
✓ Check for corrupted CSV data
```

**"Dashboard won't start"**
```bash
pip install --upgrade streamlit
streamlit run app.py
```

---

## 📋 Retrain Recommendations

Dashboard shows **one primary issue** (prioritized):

1. **Model too old** → Retrain regularly
2. **Data leakage detected** → Remove problematic features
3. **Poor data quality** → Clean data first
4. **Data drift detected** → Adapt to new distribution
5. **Label noise found** → Review inconsistencies
6. **No issues** → Keep current model

---

## 🚀 Technical Stack

| Component | Library | Version |
|-----------|---------|---------|
| Dashboard | Streamlit | 1.28.1 |
| Data Processing | Pandas | 2.1.1 |
| ML | Scikit-Learn | 1.3.1 |
| Stats | SciPy | 1.11.3 |
| Visualization | Matplotlib | 3.8.1 |
| Numerics | NumPy | 1.24.3 |

**Typical Performance:**
- Full monitoring: ~500ms
- Model retraining: ~1-5 seconds (10K rows)
- All checks: < 2 seconds combined

---

## ✨ Design Principles

✅ **Pure Functions**: No classes, no state mutations
✅ **Type Hints**: All functions fully typed
✅ **Minimal Dependencies**: Only essential libraries
✅ **Comprehensive Monitoring**: Catches 5 distinct issues
✅ **Production-Ready**: Clean, optimized codebase
✅ **Easy to Extend**: Add new projects by editing app.py
