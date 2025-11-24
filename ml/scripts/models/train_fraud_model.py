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
METADATA_DIR = '../../artifacts/metadata'
PLOTS_DIR = '../../plots'
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

def engineer_fraud_features(df):
    """
    Engineer features specifically for fraud detection

    Fraud indicators we want to capture:
    1. Exaggerated amounts
    2. Duplicate patterns
    3. Staged incident indicators
    4. Identity theft signals
    5. Out-of-pattern behavior
    """
    print("\nEngineering fraud detection features...")

    df = df.copy()

    # -- Amount-based features (exaggeration indicators)
    df['claim_amount_ratio'] = df['claim_amount'] / df['coverage_amount']
    df['claim_amount_log'] = np.log1p(df['claim_amount'])
    df['is_extreme_amount'] = (df['claim_amount_ratio'] > 0.7).astype(int)

    ## Amount compared to user's average
    df['amount_vs_user_avg'] = df['claim_amount'] / (df['user_avg_claim_amount'] + 1)
    df['is_amount_spike'] = (df['amount_vs_user_avg'] > 2.0).astype(int)

    ## Round number indicator (staged incidents often have round amounts)
    df['is_round_amount'] = (df['claim_amount'] % 1000 == 0).astype(int)

    # -- Duplicate/rapid succession indicators
    df['days_since_last_claim_filled'] = df['days_since_last_claim'].fillna(999)
    df['is_rapid_succession_numeric'] = df['is_rapid_succession'].astype(int)
    df['is_very_rapid'] = (df['days_since_last_claim_filled'] < 3).astype(int)

    ## Claim frequency intensity
    df['claims_per_month'] = df['claims_last_30_days']
    df['claims_per_quarter'] = df['claims_last_90_days']
    df['is_frequent_claimer'] = (df['claims_last_30_days'] > 2).astype(int)

    # -- Identity theft indicators
    df['is_geolocation_anomaly_numeric'] = df['is_geolocation_anomaly'].astype(int)
    df['geolocation_risk'] = np.where(
        df['geolocation_distance_km'] > 1000, 1.0,
        np.where(df['geolocation_distance_km'] > 500, 0.7,
        np.where(df['geolocation_distance_km'] > 100, 0.3, 0.0))
    )

    df['is_untrusted_device'] = (~df['is_trusted_device']).astype(int)
    df['device_risk'] = 1.0 - df['device_trust_score']

    ## Combined identity theft signal
    df['identity_theft_signal'] = (
        df['is_geolocation_anomaly_numeric'] * 0.6 +
        df['is_untrusted_device'] * 0.4
    )

    # -- Staged incident indicators
    df['has_few_documents'] = (df['supporting_documents_count'] < 2).astype(int)
    df['is_rushed_form'] = (df['form_fill_time_seconds'] < 180).astype(int)
    df['is_unusual_time_numeric'] = df['is_unusual_time'].astype(int)

    ## Staged incident composite score
    df['staged_signal'] = (
        df['is_round_amount'] * 0.3 +
        df['has_few_documents'] * 0.4 +
        df['is_rushed_form'] * 0.3
    )

    # -- Out-of-pattern indicators
    df['is_new_account'] = (df['account_age_days'] < 180).astype(int)
    df['claim_frequency_anomaly'] = (df['user_total_claims'] > df['account_age_days'] / 60).astype(int)

    ## Session behavior anomalies
    df['form_fill_time_minutes'] = df['form_fill_time_seconds'] / 60.0
    df['is_suspicious_session'] = (
        (df['form_fill_time_seconds'] < 120) |
        (df['form_fill_time_seconds'] > 1200)
    ).astype(int)

    ## Time pattern anomalies
    df['is_late_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 5)).astype(int)
    df['is_weekend'] = (df['transaction_day_of_week'].isin([5, 6])).astype(int)

    # -- Encode categorical features
    le_policy = LabelEncoder()
    df['policy_type_encoded'] = le_policy.fit_transform(df['policy_type'])

    le_claim = LabelEncoder()
    df['claim_type_encoded'] = le_claim.fit_transform(df['claim_type'])

    le_risk = LabelEncoder()
    df['risk_category_encoded'] = le_risk.fit_transform(df['risk_category'])

    le_status = LabelEncoder()
    df['claim_status_encoded'] = le_status.fit_transform(df['claim_status'])

    print(f"Engineered {len(df.columns)} total features")

    return df, {
        'policy_type': le_policy,
        'claim_type': le_claim,
        'risk_category': le_risk,
        'claim_status': le_status
    }


def select_fraud_features():
    """Select features for fraud detection"""
    features = [
        # Amount features (exaggeration)
        'claim_amount_log',
        'claim_amount_ratio',
        'is_extreme_amount',
        'amount_vs_user_avg',
        'is_amount_spike',
        'is_round_amount',

        # Duplicate/frequency features
        'days_since_last_claim_filled',
        'is_rapid_succession_numeric',
        'is_very_rapid',
        'claims_per_month',
        'claims_per_quarter',
        'is_frequent_claimer',
        'user_total_claims',

        # Identity theft features
        'geolocation_distance_km',
        'geolocation_risk',
        'is_geolocation_anomaly_numeric',
        'device_trust_score',
        'device_risk',
        'is_untrusted_device',
        'identity_theft_signal',

        # Staged incident features
        'supporting_documents_count',
        'has_few_documents',
        'form_fill_time_seconds',
        'form_fill_time_minutes',
        'is_rushed_form',
        'staged_signal',

        # Out-of-pattern features
        'account_age_days',
        'is_new_account',
        'claim_frequency_anomaly',
        'session_duration_seconds',
        'is_suspicious_session',

        # Time features
        'transaction_hour',
        'transaction_day_of_week',
        'is_unusual_time_numeric',
        'is_late_night',
        'is_weekend',

        # User/policy features
        'premium_amount',
        'coverage_amount',
        'user_avg_claim_amount',
        'policy_type_encoded',
        'claim_type_encoded',
        'risk_category_encoded',
        'claim_status_encoded'
    ]

    return features


def train_isolation_forest(X_train, contamination=0.18):
    """Train Isolation Forest for anomaly detection"""
    print("\n" + "=" * 70)
    print("TRAINING ISOLATION FOREST (Unsupervised Anomaly Detection)")
    print("=" * 70)

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )

    print(f"Training with contamination={contamination}...")
    iso_forest.fit(X_train)

    # Get anomaly scores
    anomaly_scores = iso_forest.decision_function(X_train)
    anomaly_predictions = iso_forest.predict(X_train)  # -1 for anomalies, 1 for normal

    n_anomalies = (anomaly_predictions == -1).sum()
    print(f"✓ Detected {n_anomalies} anomalies in training set ({n_anomalies / len(X_train) * 100:.2f}%)")

    return iso_forest


def train_supervised_classifier(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier for fraud detection"""
    print("\n" + "=" * 70)
    print("TRAINING XGBOOST CLASSIFIER (Supervised)")
    print("=" * 70)

    print(f"Training set: {len(X_train)} samples")
    print(f"  - Fraudulent: {y_train.sum()} ({y_train.mean() * 100:.2f}%)")
    print(f"  - Legitimate: {(~y_train).sum()} ({(~y_train).mean() * 100:.2f}%)")

    # The training data is imbalanced, so we would handle the uneven distribution
    # Handle class imbalance with SMOTE
    print("\nApplying SMOTE to balance classes...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE: {len(X_train_balanced)} samples")
    print(f"  - Fraudulent: {y_train_balanced.sum()} ({y_train_balanced.mean() * 100:.2f}%)")
    print(f"  - Legitimate: {(~y_train_balanced).sum()} ({(~y_train_balanced).mean() * 100:.2f}%)")

    # Train initial model
    print("\nTraining initial XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=RANDOM_STATE,
        eval_metric='auc',
        use_label_encoder=False,
        # scale_pos_weight=(~y_train).sum() / y_train.sum()  # Handle imbalance
    )

    # Cross-validation
    cv_scores = cross_val_score(
        xgb_model, X_train_balanced, y_train_balanced,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    param_grid = {
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 300],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'min_child_weight': [1, 3]
    }

    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_balanced, y_train_balanced)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")

    # Final model
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 70)
    print("FRAUD DETECTION MODEL EVALUATION")
    print("=" * 70)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Legitimate', 'Fraudulent'],
        digits=4
    ))

    # Detailed metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nDetailed Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nTrue Negatives: {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}")
    print(f"True Positives: {cm[1, 1]}")

    return best_model, y_pred, y_pred_proba, smote


def plot_fraud_metrics(y_test, y_pred_proba):
    """Plot ROC curve and Precision-Recall curve"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve - Fraud Detection', fontsize=14)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)

    axes[1].plot(recall, precision, color='darkgreen', lw=2, label='Precision-Recall curve')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=14)
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/fraud_roc_pr_curves.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ ROC and PR curves saved to {PLOTS_DIR}/fraud_roc_pr_curves.png")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance for fraud detection"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='rocket')
    plt.title(f'Top {top_n} Most Important Features for Fraud Detection', fontsize=14, pad=15)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/fraud_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved to {PLOTS_DIR}/fraud_feature_importance.png")
    plt.close()

    return importance_df


def plot_confusion_matrix_fraud(y_test, y_pred):
    """Plot confusion matrix for fraud detection"""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Reds',
        xticklabels=['Legitimate', 'Fraudulent'],
        yticklabels=['Legitimate', 'Fraudulent'],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Fraud Detection - Confusion Matrix', fontsize=14, pad=15)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/fraud_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {PLOTS_DIR}/fraud_confusion_matrix.png")
    plt.close()


def save_fraud_model_artifacts(iso_forest, xgb_model, scaler, smote, label_encoders, feature_names, metadata):
    """Save all fraud detection model artifacts"""
    print("\nSaving fraud detection model artifacts...")

    # Save Isolation Forest
    iso_path = f'{MODEL_OUTPUT_DIR}/isolation_forest.pkl'
    joblib.dump(iso_forest, iso_path)
    print(f"✓ Isolation Forest saved to {iso_path}")

    # Save XGBoost classifier
    xgb_path = f'{MODEL_OUTPUT_DIR}/fraud_classifier.pkl'
    joblib.dump(xgb_model, xgb_path)
    print(f"✓ XGBoost classifier saved to {xgb_path}")

    # Save scaler
    scaler_path = f'{MODEL_OUTPUT_DIR}/fraud_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")

    # Save SMOTE (for reference, not used in production)
    smote_path = f'{MODEL_OUTPUT_DIR}/fraud_smote.pkl'
    joblib.dump(smote, smote_path)
    print(f"✓ SMOTE saved to {smote_path}")

    # Save label encoders
    encoders_path = f'{MODEL_OUTPUT_DIR}/fraud_label_encoders.pkl'
    joblib.dump(label_encoders, encoders_path)
    print(f"✓ Label encoders saved to {encoders_path}")

    # Save feature names
    features_path = f'{METADATA_DIR}/fraud_feature_names.json'
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"✓ Feature names saved to {features_path}")

    # Save metadata
    metadata_path = f'{METADATA_DIR}/fraud_model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")


def main():
    """Main execution function"""
    print("=" * 70)
    print("FRAUD DETECTION MODEL TRAINING")
    print("=" * 70)

    # Load data
    df = load_and_prepare_data()

    # Engineer features
    df, label_encoders = engineer_fraud_features(df)

    # Save feature engineered dataset
    df.to_csv("../../data/processed/fraud_engineered_v1.csv", index=False)
    print(f"✓ Saved fraud engineered dataset to ml/data/processed/fraud_engineered_v1.csv")

    # Select features
    feature_names = select_fraud_features()
    print(f"\nSelected {len(feature_names)} features for fraud detection")

    # Prepare data
    X = df[feature_names]
    y = df['is_fraudulent'].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Isolation Forest (unsupervised)
    contamination = y_train.mean()  # Use actual fraud rate
    iso_forest = train_isolation_forest(X_train_scaled, contamination)

    # Get anomaly scores for test set
    anomaly_scores_test = iso_forest.decision_function(X_test_scaled)
    anomaly_pred_test = iso_forest.predict(X_test_scaled)
    anomaly_pred_binary = (anomaly_pred_test == -1).astype(int)

    print(f"\nIsolation Forest Test Set Performance:")
    print(f"  Detected anomalies: {anomaly_pred_binary.sum()} ({anomaly_pred_binary.mean() * 100:.2f}%)")
    print(f"  True fraud rate: {y_test.mean() * 100:.2f}%")

    # Train supervised classifier
    xgb_model, y_pred, y_pred_proba, smote = train_supervised_classifier(
        X_train_scaled, y_train,
        X_test_scaled, y_test
    )

    # Plot metrics
    plot_fraud_metrics(y_test, y_pred_proba)

    # Plot feature importance
    importance_df = plot_feature_importance(xgb_model, feature_names)

    # Plot confusion matrix
    plot_confusion_matrix_fraud(y_test, y_pred)

    # Calculate final metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Prepare metadata
    metadata = {
        'model_type': 'Ensemble (Isolation Forest + XGBoost)',
        'objective': 'Binary Fraud Classification',
        'training_date': datetime.now().isoformat(),
        'n_features': len(feature_names),
        'n_training_samples': len(X_train),
        'n_test_samples': len(X_test),
        'fraud_rate_train': float(y_train.mean()),
        'fraud_rate_test': float(y_test.mean()),
        'smote_applied': True,
        'test_metrics': {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc)
        },
        'isolation_forest': {
            'contamination': float(contamination),
            'n_estimators': 100
        },
        'top_10_features': importance_df.head(10)['feature'].tolist()
    }

    # Save all artifacts
    save_fraud_model_artifacts(
        iso_forest,
        xgb_model,
        scaler,
        smote,
        label_encoders,
        feature_names,
        metadata
    )

    print("\n" + "=" * 70)
    print("FRAUD DETECTION MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel Performance Summary:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"\nTop 5 Important Features:")
    for i, feature in enumerate(importance_df.head(5)['feature'], 1):
        print(f"  {i}. {feature}")


if __name__ == "__main__":
    main()
