import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
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
PLOTS_DIR = '../../plots'
DATA_PATH = '../../data/synthetic/insurance_claims_v1_full.csv'

def load_and_prepare_data():
    """Load and prepare data for risk scoring model"""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Convert datetime columns
    df['submitted_at'] = pd.to_datetime(df['submitted_at'])
    df['incident_date'] = pd.to_datetime(df['incident_date'])
    df['policy_start_date'] = pd.to_datetime(df['policy_start_date'])

    print(f"Loaded {len(df)} records")
    return df

def engineer_risk_features(df):
    """
    Engineer features specifically for risk scoring

    Risk factors we want to capture:
    1. High claim amount relative to coverage
    2. Multiple recent claims
    3. Rapid succession of claims
    4. Unusual geolocation
    5. New/untrusted device
    6. Unusual time patterns
    7. Behavioral anomalies
    """
    print("\nEngineering risk features...")

    # Make a copy to avoid modifying original
    df = df.copy()

    # -- Claim amount features
    df['claim_amount_ratio'] = df['claim_amount'] / df['coverage_amount']
    df['claim_amount_log'] = np.log1p(df['claim_amount'])
    df['is_high_claim'] = (df['claim_amount_ratio'] > 0.5).astype(int)

    # -- Claim frequency features
    df['has_recent_claims'] = (df['claims_last_30_days'] > 0).astype(int)
    df['has_multiple_recent_claims'] = (df['claims_last_30_days'] > 1).astype(int)
    df['claims_intensity'] = df['claims_last_30_days'] / 30.0  # Claims per day

    # -- Rapid succession features
    df['days_since_last_claim_filled'] = df['days_since_last_claim'].fillna(999)
    df['is_rapid_succession_numeric'] = df['is_rapid_succession'].astype(int)

    # -- Geolocation features
    df['geolocation_risk_score'] = np.where(
        df['geolocation_distance_km'] > 500, 1.0,
        np.where(df['geolocation_distance_km'] > 100, 0.5, 0.0)
    )
    df['is_geolocation_anomaly_numeric'] = df['is_geolocation_anomaly'].astype(int)

    # -- Device trust features
    df['is_untrusted_device'] = (~df['is_trusted_device']).astype(int)
    df['device_risk_score'] = 1.0 - df['device_trust_score']

    # -- Time-based features
    df['is_unusual_time_numeric'] = df['is_unusual_time'].astype(int)
    df['is_business_hours'] = ((df['transaction_hour'] >= 9) & (df['transaction_hour'] <= 17)).astype(int)
    df['is_late_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 5)).astype(int)

    # -- Session behavior features
    df['form_fill_time_minutes'] = df['form_fill_time_seconds'] / 60.0
    df['is_rushed_form'] = (df['form_fill_time_seconds'] < 180).astype(int)
    df['is_suspicious_session'] = (
        (df['form_fill_time_seconds'] < 120) |
        (df['form_fill_time_seconds'] > 1200)
    ).astype(int)

    # -- User history features
    df['account_age_years'] = df['account_age_days'] / 365.0
    df['is_new_account'] = (df['account_age_days'] < 180).astype(int)

    # -- Document count feature
    df['has_few_documents'] = (df['supporting_documents_count'] < 2).astype(int)

    # -- Policy features
    df['premium_ratio'] = df['premium_amount'] / df['coverage_amount']

    # Encode categorical features
    le_policy = LabelEncoder()
    df['policy_type_encoded'] = le_policy.fit_transform(df['policy_type'])

    le_claim = LabelEncoder()
    df['claim_type_encoded'] = le_claim.fit_transform(df['claim_type'])

    le_risk = LabelEncoder()
    df['risk_category_encoded'] = le_risk.fit_transform(df['risk_category'])

    print(f"Engineered {len(df.columns)} total features")

    return df, le_policy, le_claim, le_risk

def create_risk_labels(df):
    """
    Create risk labels based on fraud probability and other factors

    Risk levels:
    - Low (0): fraud_confidence < 0.3 AND no suspicious patterns
    - Medium (1): fraud_confidence 0.3-0.65 OR some suspicious patterns
    - High (2): fraud_confidence > 0.65 OR multiple suspicious patterns
    """
    print("\nCreating risk labels...")

    # Calculate composite risk score
    risk_score = (
        df['fraud_confidence_score'] * 0.4 +  # Fraud probability
        df['claim_amount_ratio'] * 0.15 +     # High claim amount
        (df['claims_last_30_days'] / 10.0) * 0.15 +  # Recent claim frequency
        df['is_geolocation_anomaly_numeric'] * 0.15 +  # Location anomaly
        df['is_untrusted_device'] * 0.10 +    # Device trust
        df['is_unusual_time_numeric'] * 0.05  # Time pattern
    )

    # Classify into risk levels
    risk_level = pd.cut(
        risk_score,
        bins=[-np.inf, 0.35, 0.65, np.inf],
        labels=[0, 1, 2]  # Low, Medium, High
    ).astype(int)

    df['risk_level'] = risk_level

    # Print distribution
    risk_dist = risk_level.value_counts().sort_index()
    print(f"\nRisk Level Distribution:")
    print(f"  Low (0): {risk_dist[0]} ({risk_dist[0]/len(df)*100:.1f}%)")
    print(f"  Medium (1): {risk_dist[1]} ({risk_dist[1]/len(df)*100:.1f}%)")
    print(f"  High (2): {risk_dist[2]} ({risk_dist[2]/len(df)*100:.1f}%)")

    return df

def select_features():
    """Select features for the risk scoring model"""
    features = [
        # Claim amount features
        'claim_amount_log',
        'claim_amount_ratio',
        'is_high_claim',

        # Claim frequency features
        'claims_last_30_days',
        'claims_last_90_days',
        'has_recent_claims',
        'has_multiple_recent_claims',
        'claims_intensity',
        'user_total_claims',

        # Temporal features
        'days_since_last_claim_filled',
        'is_rapid_succession_numeric',

        # Geolocation features
        'geolocation_distance_km',
        'geolocation_risk_score',
        'is_geolocation_anomaly_numeric',

        # Device features
        'device_trust_score',
        'device_risk_score',
        'is_untrusted_device',

        # Time features
        'transaction_hour',
        'transaction_day_of_week',
        'is_unusual_time_numeric',
        'is_business_hours',
        'is_late_night',
        'is_weekend',

        # Session behavior
        'session_duration_seconds',
        'form_fill_time_minutes',
        'is_rushed_form',
        'is_suspicious_session',
        'pages_visited',

        # User features
        'account_age_days',
        'account_age_years',
        'is_new_account',
        'user_avg_claim_amount',

        # Document features
        'supporting_documents_count',
        'has_few_documents',

        # Policy features
        'premium_amount',
        'coverage_amount',
        'premium_ratio',
        'policy_type_encoded',
        'claim_type_encoded',
        'risk_category_encoded'
    ]

    return features

def train_risk_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model for risk classification."""
    print("\nTraining Risk Scoring Model...")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

     # Define parameter grid for tuning
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'min_child_weight': [1, 3, 5]
    }

    # Initial model with default parameters
    print("\nTraining initial model...")
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss',
        use_label_encoder=False
    )

    # Cross-validation
    cv_scores = cross_val_score(
        xgb_model, X_train, y_train,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Grid search for hyperparameter tuning (using smaller grid for speed)
    print("\nPerforming hyperparameter tuning...")
    param_grid_small = {
        'max_depth': [6, 8],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [200, 300],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    grid_search = GridSearchCV(
        xgb_model,
        param_grid_small,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    # Train final model with best parameters
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)

    print("\n" + "="*70)
    print("RISK MODEL EVALUATION")
    print("="*70)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Low Risk', 'Medium Risk', 'High Risk']
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Calculate accuracy per class
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class Accuracy:")
    for i, acc in enumerate(class_accuracy):
        risk_name = ['Low', 'Medium', 'High'][i]
        print(f"  {risk_name} Risk: {acc:.4f}")

    # Overall accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # Calculate AUC for each class (one-vs-rest)
    print("\nAUC Scores (One-vs-Rest):")
    for i in range(3):
        y_test_binary = (y_test == i).astype(int)
        auc = roc_auc_score(y_test_binary, y_pred_proba[:, i])
        risk_name = ['Low', 'Medium', 'High'][i]
        print(f"  {risk_name} Risk: {auc:.4f}")

    return best_model, y_pred, y_pred_proba

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Most Important Features for Risk Scoring', fontsize=14, pad=15)

    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()

    plt.savefig(f'{PLOTS_DIR}/risk_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Feature importance plot saved to {PLOTS_DIR}/risk_feature_importance.png")
    plt.close()

    return importance_df

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Low', 'Medium', 'High'],
        yticklabels=['Low', 'Medium', 'High'],
        cbar_kws={'label': 'Count'}
    )

    plt.title('Risk Model - Confusion Matrix', fontsize=14, pad=15)
    plt.xlabel('Predicted Risk Level', fontsize=12)
    plt.ylabel('True Risk Level', fontsize=12)
    plt.tight_layout()

    plt.savefig(f'{PLOTS_DIR}/risk_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {PLOTS_DIR}/risk_confusion_matrix.png")
    plt.close()

def save_model_artifacts(model, scaler, label_encoders, feature_names, metadata):
    """Save model and associated artifacts"""
    print("\nSaving model artifacts...")

    # Save model
    model_path = f'{MODEL_OUTPUT_DIR}/risk_model.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Model saved to {model_path}")

    # Save scaler
    scaler_path = f'{MODEL_OUTPUT_DIR}/risk_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")

    # Save label encoders
    encoders_path = f'{MODEL_OUTPUT_DIR}/risk_label_encoders.pkl'
    joblib.dump(label_encoders, encoders_path)
    print(f"✓ Label encoders saved to {encoders_path}")

    # Save feature names
    features_path = f'{MODEL_OUTPUT_DIR}/risk_feature_names.json'
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"✓ Feature names saved to {features_path}")

    # Save metadata
    metadata_path = f'{MODEL_OUTPUT_DIR}/risk_model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")


def main():
    """Main execution function"""
    print("="*70)
    print("RISK SCORING MODEL TRAINING")
    print("="*70)

    # Load data
    df = load_and_prepare_data()

    # Engineer features
    df, le_policy, le_claim, le_risk = engineer_risk_features(df)

    # Create risk labels
    df = create_risk_labels(df)

    # Save feature engineered dataset
    df.to_csv("../../data/processed/risk_engineered_v1.csv", index=False)
    print(f"✓ Saved risk engineered dataset to ml/data/processed/risk_engineered_v1.csv")

    # Select features
    feature_names = select_features()
    print(f"\nSelected {len(feature_names)} features for training")

    # Prepare train/test split
    X = df[feature_names]
    y = df['risk_level']

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

    # Train model
    model, y_pred, y_pred_proba = train_risk_model(
        X_train_scaled, y_train,
        X_test_scaled, y_test
    )

    # Plot feature importance
    importance_df = plot_feature_importance(model, feature_names)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # Prepare metadata
    metadata = {
        'model_type': 'XGBoost Classifier',
        'objective': 'Multi-class Risk Classification (Low/Medium/High)',
        'training_date': datetime.now().isoformat(),
        'n_features': len(feature_names),
        'n_training_samples': len(X_train),
        'n_test_samples': len(X_test),
        'test_accuracy': float((y_pred == y_test).mean()),
        'risk_level_mapping': {
            '0': 'Low',
            '1': 'Medium',
            '2': 'High'
        },
        'top_10_features': importance_df.head(10)['feature'].tolist()
    }

    # Save artifacts
    label_encoders = {
        'policy_type': le_policy,
        'claim_type': le_claim,
        'risk_category': le_risk
    }

    save_model_artifacts(
        model,
        scaler,
        label_encoders,
        feature_names,
        metadata
    )

    print("\n" + "="*70)
    print("RISK MODEL TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel achieves {metadata['test_accuracy']*100:.2f}% accuracy on test set")
    print(f"\nTop 5 Important Features:")
    for i, feature in enumerate(importance_df.head(5)['feature'], 1):
        print(f"  {i}. {feature}")


if __name__ == "__main__":
    main()
