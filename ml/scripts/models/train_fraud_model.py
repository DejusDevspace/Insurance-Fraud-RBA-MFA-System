import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_OUTPUT_DIR = '../../artifacts/models'
DATA_PATH = '../../data/synthetic/insurance_claims_v1_full.csv'


def load_and_prepare_data():
    """Load and prepare data for fraud detection"""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Convert date columns
    df['submitted_at'] = pd.to_datetime(df['submitted_at'])
    df['incident_date'] = pd.to_datetime(df['incident_date'])
    df['policy_start_date'] = pd.to_datetime(df['policy_start_date'])

    print(f"Loaded {len(df)} records")
    print(f"Fraud rate: {df['is_fraudulent'].mean() * 100:.2f}%")

    return df
