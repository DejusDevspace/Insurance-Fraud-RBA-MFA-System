import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

from app.ml.model_loader import get_model_loader
from app.config import settings

logger = logging.getLogger(__name__)


class FraudPredictor:
    """
    Fraud detection engine
    Uses ensemble of Isolation Forest (unsupervised) and XGBoost (supervised)
    """

    def __init__(self):
        self.model_loader = get_model_loader()
        self.isolation_forest = self.model_loader.get_isolation_forest()
        self.classifier = self.model_loader.get_fraud_classifier()
        self.scaler = self.model_loader.get_fraud_scaler()
        self.feature_names = self.model_loader.get_fraud_features()
        self.encoders = self.model_loader.get_fraud_encoders()

    def detect_fraud(self, features: Dict) -> Tuple[bool, float, str, float, str]:
        """
        Detect fraud using ensemble approach

        Args:
            features: Dictionary of engineered features

        Returns:
            Tuple of (is_suspicious, fraud_probability, predicted_fraud_type,
                     anomaly_score, model_used)
        """
        try:
            # Extract and scale features
            feature_vector = self._extract_feature_vector(features)
            feature_vector_scaled = self.scaler.transform(feature_vector)

            # Anomaly detection (Isolation Forest)
            anomaly_score = self.isolation_forest.decision_function(feature_vector_scaled)[0]
            is_anomaly = self.isolation_forest.predict(feature_vector_scaled)[0] == -1

            # Supervised classification (XGBoost)
            fraud_probability = self.classifier.predict_proba(feature_vector_scaled)[0][1]

            # Ensemble decision
            is_suspicious = (
                    is_anomaly or
                    fraud_probability > settings.FRAUD_PROBABILITY_THRESHOLD
            )

            # Detect fraud type if suspicious
            predicted_fraud_type = None
            if is_suspicious:
                predicted_fraud_type = self._detect_fraud_type(features, fraud_probability)

            # Determine model used
            if is_anomaly and fraud_probability > settings.FRAUD_PROBABILITY_THRESHOLD:
                model_used = "ensemble"
            elif is_anomaly:
                model_used = "isolation_forest"
            else:
                model_used = "supervised_classifier"

            logger.info(
                f"Fraud detection: suspicious={is_suspicious}, "
                f"probability={fraud_probability:.4f}, type={predicted_fraud_type}"
            )

            return is_suspicious, fraud_probability, predicted_fraud_type, anomaly_score, model_used

        except Exception as e:
            logger.error(f"Fraud detection failed: {str(e)}")
            raise

    def _extract_feature_vector(self, features: Dict) -> np.ndarray:
        """
        Extract feature vector in the correct order matching training

        Args:
            features: Dictionary of features

        Returns:
            numpy array of features
        """
        feature_values = []

        for feature_name in self.feature_names:
            value = features.get(feature_name, 0)

            # Handle missing values
            if value is None:
                value = 0

            feature_values.append(value)

        return np.array(feature_values).reshape(1, -1)

    def _detect_fraud_type(self, features: Dict, fraud_probability: float) -> str:
        """
        Detect the type of fraud based on feature patterns

        Args:
            features: Dictionary of features
            fraud_probability: Fraud probability from classifier

        Returns:
            Predicted fraud type
        """
        # Exaggerated claim (high amount spike)
        if features.get('is_amount_spike', 0) == 1 or features.get('amount_vs_user_avg', 1) > 2.5:
            return "exaggerated"

        # Duplicate claim (very rapid succession)
        if features.get('is_very_rapid', 0) == 1 or features.get('days_since_last_claim_filled', 999) < 3:
            return "duplicate"

        # Staged incident (round amount, few documents)
        if features.get('is_round_amount', 0) == 1 and features.get('has_few_documents', 0) == 1:
            return "staged"

        # Identity theft (location + device anomalies)
        if (features.get('is_geolocation_anomaly_numeric', 0) == 1 and
                features.get('is_untrusted_device', 0) == 1):
            return "identity_theft"

        # Out-of-pattern (frequent claims, new account)
        if features.get('is_frequent_claimer', 0) == 1 or features.get('claim_frequency_anomaly', 0) == 1:
            return "out_of_pattern"

        # Default
        return "unknown"

    def get_confidence_level(self, fraud_probability: float) -> str:
        """
        Get confidence level for fraud prediction

        Args:
            fraud_probability: Fraud probability (0-1)

        Returns:
            Confidence level (low, medium, high)
        """
        if fraud_probability < 0.5:
            return "low"
        elif fraud_probability < settings.FRAUD_HIGH_CONFIDENCE_THRESHOLD:
            return "medium"
        else:
            return "high"
