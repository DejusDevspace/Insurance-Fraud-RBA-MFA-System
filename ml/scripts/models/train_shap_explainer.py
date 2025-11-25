import pandas as pd
import numpy as np
import joblib
import shap
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
MODEL_OUTPUT_DIR = '../../artifacts/models'
RISK_DATA_PATH = '../../data/processed/risk_engineered_v1.csv'
FRAUD_DATA_PATH = '../../data/processed/fraud_engineered_v1.csv'
PLOTS_DIR = '../../plots'
METADATA_DIR = '../../artifacts/metadata'
N_SAMPLES_BACKGROUND = 100  # Number of samples for SHAP background
N_SAMPLES_EXPLAIN = 10  # Number of samples to explain for testing


def load_models_and_data():
    """Load trained models and test data"""
    print("Loading trained models and data...")

    # Load models
    risk_model = joblib.load(f'{MODEL_OUTPUT_DIR}/risk_model.pkl')
    fraud_model = joblib.load(f'{MODEL_OUTPUT_DIR}/fraud_classifier.pkl')

    # Load scalers
    risk_scaler = joblib.load(f'{MODEL_OUTPUT_DIR}/risk_scaler.pkl')
    fraud_scaler = joblib.load(f'{MODEL_OUTPUT_DIR}/fraud_scaler.pkl')

    # Load feature names
    with open(f'{METADATA_DIR}/risk_feature_names.json', 'r') as f:
        risk_features = json.load(f)

    with open(f'{METADATA_DIR}/fraud_feature_names.json', 'r') as f:
        fraud_features = json.load(f)

    # Load data
    risk_df = pd.read_csv(RISK_DATA_PATH)
    fraud_df = pd.read_csv(FRAUD_DATA_PATH)

    print("✓ Models and data loaded successfully")

    return risk_model, fraud_model, risk_scaler, fraud_scaler, risk_features, fraud_features, risk_df, fraud_df


def create_risk_explainer(model, X_background, feature_names):
    """Create SHAP explainer for risk model"""
    print("\n" + "=" * 70)
    print("CREATING SHAP EXPLAINER FOR RISK MODEL")
    print("=" * 70)

    print(f"Creating TreeExplainer with {len(X_background)} background samples...")
    explainer = shap.TreeExplainer(
        model,
        X_background,
        feature_names=feature_names,
        model_output='raw'
    )

    print("✓ Risk explainer created")

    return explainer


def create_fraud_explainer(model, X_background, feature_names):
    """Create SHAP explainer for fraud model"""
    print("\n" + "=" * 70)
    print("CREATING SHAP EXPLAINER FOR FRAUD MODEL")
    print("=" * 70)

    print(f"Creating TreeExplainer with {len(X_background)} background samples...")
    explainer = shap.TreeExplainer(
        model,
        X_background,
        feature_names=feature_names
    )

    print("✓ Fraud explainer created")

    return explainer


def test_risk_explainer(explainer, X_test, feature_names, n_samples=5):
    """Test risk explainer and generate sample explanations"""
    print(f"\nTesting risk explainer with {n_samples} samples...")

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test[:n_samples])

    # For multi-class, shap_values is a list of arrays (one per class)
    # We'll use the values for the predicted class

    print("✓ SHAP values calculated")

    # Generate waterfall plots for all three risk classes
    print("\nGenerating waterfall plots for all risk classes...")

    risk_class_names = ['Low Risk', 'Medium Risk', 'High Risk']

    if isinstance(shap_values, list):
        # List format
        for class_idx, class_name in enumerate(risk_class_names):
            sample_shap_values = shap_values[class_idx][0]
            base_value = explainer.expected_value[class_idx]

            explanation = shap.Explanation(
                values=sample_shap_values,
                base_values=base_value,
                data=X_test[0],
                feature_names=feature_names
            )

            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(explanation, show=False)
            plt.title(f'SHAP Explanation - {class_name}')
            plt.tight_layout()
            plt.savefig(f'{PLOTS_DIR}/risk_shap_waterfall_{class_name.lower().replace(" ", "_")}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Waterfall plot for {class_name} saved")
    else:
        # 3D array format
        for class_idx, class_name in enumerate(risk_class_names):
            sample_shap_values = shap_values[0, :, class_idx]
            base_value = explainer.expected_value[class_idx]

            explanation = shap.Explanation(
                values=sample_shap_values,
                base_values=base_value,
                data=X_test[0],
                feature_names=feature_names
            )

            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(explanation, show=False)
            plt.title(f'SHAP Explanation - {class_name}')
            plt.tight_layout()
            plt.savefig(f'{PLOTS_DIR}/risk_shap_waterfall_{class_name.lower().replace(" ", "_")}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Waterfall plot for {class_name} saved")

    return shap_values


def test_fraud_explainer(explainer, X_test, feature_names, n_samples=5):
    """Test fraud explainer and generate sample explanations"""
    print(f"\nTesting fraud explainer with {n_samples} samples...")

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test[:n_samples])

    print("✓ SHAP values calculated")

    # Generate waterfall plot for first sample
    print("\nGenerating sample waterfall plot...")

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test[0],
        feature_names=feature_names
    )

    # Save waterfall plot
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/fraud_shap_waterfall_sample.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Waterfall plot saved to {PLOTS_DIR}/fraud_shap_waterfall_sample.png")

    # Generate summary plot
    print("\nGenerating summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[:n_samples], feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/fraud_shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Summary plot saved to {PLOTS_DIR}/fraud_shap_summary.png")

    return shap_values


def create_explanation_function(explainer, feature_names, model=None):
    """
    Create a function that generates explanations for new samples
    This will be used in the backend API
    """

    def explain_prediction(X_sample, top_n=5):
        """
        Generate explanation for a single prediction

        Args:
            X_sample: Feature vector (1D array or 2D array with single row)
            top_n: Number of top features to return

        Returns:
            dict: Explanation with SHAP values and top features
        """
        # Ensure X_sample is 2D
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)

        # Handle multi-class output
        if isinstance(shap_values, list):
            # List format: [array for class 0, array for class 1, array for class 2]
            # Get predicted class
            if model is not None:
                predicted_class = np.argmax(model.predict(X_sample)[0])
            else:
                # Fallback: use class with highest sum of SHAP values
                predicted_class = np.argmax([sv[0].sum() for sv in shap_values])

            shap_vals = shap_values[predicted_class][0]  # Shape: (n_features,)
            base_value = explainer.expected_value[predicted_class]

        elif len(shap_values.shape) == 3:
            # 3D array format: (n_samples, n_features, n_classes)
            # Get predicted class
            if model is not None:
                predicted_class = np.argmax(model.predict(X_sample)[0])
            else:
                # Fallback: use class with highest sum of SHAP values
                predicted_class = np.argmax(shap_values[0].sum(axis=0))

            shap_vals = shap_values[0, :, predicted_class]  # Shape: (n_features,)
            base_value = explainer.expected_value[predicted_class]

        else:
            # Binary classification or already 1D
            if len(shap_values.shape) == 2:
                shap_vals = shap_values[0]  # Shape: (n_features,)
            else:
                shap_vals = shap_values

            base_value = explainer.expected_value if np.isscalar(explainer.expected_value) else \
            explainer.expected_value[0]

        # Create feature importance dictionary
        feature_importance = {
            feature: float(shap_val)
            for feature, shap_val in zip(feature_names, shap_vals)
        }

        # Get top N features by absolute SHAP value
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        explanation = {
            'shap_values': feature_importance,
            'top_features': [
                {
                    'feature': feature,
                    'shap_value': shap_val,
                    'contribution': 'increases' if shap_val > 0 else 'decreases',
                    'magnitude': abs(shap_val)
                }
                for feature, shap_val in top_features
            ],
            'base_value': float(base_value)
        }

        return explanation

    return explain_prediction


def save_explainer_artifacts(risk_explainer, fraud_explainer):
    """Save SHAP explainers (functions will be recreated on load)"""
    print("\nSaving SHAP explainer artifacts...")

    # Save explainers
    risk_explainer_path = f'{MODEL_OUTPUT_DIR}/risk_shap_explainer.pkl'
    joblib.dump(risk_explainer, risk_explainer_path)
    print(f"✓ Risk SHAP explainer saved to {risk_explainer_path}")

    fraud_explainer_path = f'{MODEL_OUTPUT_DIR}/fraud_shap_explainer.pkl'
    joblib.dump(fraud_explainer, fraud_explainer_path)
    print(f"✓ Fraud SHAP explainer saved to {fraud_explainer_path}")

    # Save metadata
    metadata = {
        'creation_date': datetime.now().isoformat(),
        'shap_version': shap.__version__,
        'explainer_type': 'TreeExplainer',
        'models': ['risk_model', 'fraud_classifier'],
        'usage': {
            'risk': 'Explains why a transaction was classified as low/medium/high risk',
            'fraud': 'Explains why a transaction was flagged as potentially fraudulent'
        },
        'note': 'Explanation functions should be recreated using create_explanation_function() when loading' # not picklable error
    }

    metadata_path = f'{METADATA_DIR}/shap_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ SHAP metadata saved to {metadata_path}")

def main():
    """Main execution function"""
    print("=" * 70)
    print("SHAP EXPLAINER TRAINING AND TESTING")
    print("=" * 70)

    # Load models and data
    risk_model, fraud_model, risk_scaler, fraud_scaler, risk_features, fraud_features, risk_df, fraud_df = load_models_and_data()

    # Prepare risk data
    print("\nPreparing risk model data...")
    X_risk = risk_df[risk_features].fillna(0)
    X_risk_scaled = risk_scaler.transform(X_risk)

    # Select background samples for risk
    background_indices_risk = np.random.choice(len(X_risk_scaled), N_SAMPLES_BACKGROUND, replace=False)
    X_risk_background = X_risk_scaled[background_indices_risk]

    # Select test samples for risk
    test_indices_risk = np.random.choice(
        [i for i in range(len(X_risk_scaled)) if i not in background_indices_risk],
        N_SAMPLES_EXPLAIN,
        replace=False
    )
    X_risk_test = X_risk_scaled[test_indices_risk]

    # Create risk explainer
    risk_explainer = create_risk_explainer(risk_model, X_risk_background, risk_features)

    # Test risk explainer
    risk_shap_values = test_risk_explainer(risk_explainer, X_risk_test, risk_features, N_SAMPLES_EXPLAIN)

    # Prepare fraud data
    print("\nPreparing fraud model data...")
    X_fraud = fraud_df[fraud_features].fillna(0)
    X_fraud_scaled = fraud_scaler.transform(X_fraud)

    # Select background samples for fraud
    background_indices_fraud = np.random.choice(len(X_fraud_scaled), N_SAMPLES_BACKGROUND, replace=False)
    X_fraud_background = X_fraud_scaled[background_indices_fraud]

    # Select test samples for fraud (prioritize fraudulent cases)
    fraud_indices = fraud_df[fraud_df['is_fraudulent'] == True].index.tolist()
    if len(fraud_indices) >= N_SAMPLES_EXPLAIN // 2:
        test_fraud_indices = np.random.choice(fraud_indices, N_SAMPLES_EXPLAIN // 2, replace=False).tolist()
        remaining = N_SAMPLES_EXPLAIN - len(test_fraud_indices)
        legit_indices = [i for i in range(len(fraud_df)) if i not in fraud_indices and i not in background_indices_fraud]
        test_legit_indices = np.random.choice(legit_indices, remaining, replace=False).tolist()
        test_indices_fraud = test_fraud_indices + test_legit_indices
    else:
        test_indices_fraud = np.random.choice(
            [i for i in range(len(X_fraud_scaled)) if i not in background_indices_fraud],
            N_SAMPLES_EXPLAIN,
            replace=False
        )

    X_fraud_test = X_fraud_scaled[test_indices_fraud]

    # Create fraud explainer
    fraud_explainer = create_fraud_explainer(fraud_model, X_fraud_background, fraud_features)

    # Test fraud explainer
    fraud_shap_values = test_fraud_explainer(fraud_explainer, X_fraud_test, fraud_features, N_SAMPLES_EXPLAIN)

    # Create explanation functions
    print("\nCreating explanation functions...")
    risk_explain_fn = create_explanation_function(risk_explainer, risk_features, risk_model)
    fraud_explain_fn = create_explanation_function(fraud_explainer, fraud_features, fraud_model)
    print("✓ Explanation functions created")

    # Test explanation functions
    print("\nTesting explanation functions...")
    risk_explanation = risk_explain_fn(X_risk_test[0])
    fraud_explanation = fraud_explain_fn(X_fraud_test[0])

    print("\nSample Risk Explanation:")
    print(f"  Base value: {risk_explanation['base_value']:.4f}")
    print(f"  Top 3 features:")
    for feat in risk_explanation['top_features'][:3]:
        print(f"    - {feat['feature']}: {feat['shap_value']:.4f} ({feat['contribution']})")

    print("\nSample Fraud Explanation:")
    print(f"  Base value: {fraud_explanation['base_value']:.4f}")
    print(f"  Top 3 features:")
    for feat in fraud_explanation['top_features'][:3]:
        print(f"    - {feat['feature']}: {feat['shap_value']:.4f} ({feat['contribution']})")

    # Save artifacts
    save_explainer_artifacts(risk_explainer, fraud_explainer)

    print("\n" + "=" * 70)
    print("SHAP EXPLAINER TRAINING COMPLETE")
    print("=" * 70)
    print("\nExplainers are ready to be integrated into the backend API")
    print("They will provide transparent, interpretable explanations for:")
    print("  1. Why a transaction was classified as a certain risk level")
    print("  2. Why a transaction was flagged as potentially fraudulent")


if __name__ == "__main__":
    main()
