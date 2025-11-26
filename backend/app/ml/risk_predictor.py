import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

from app.ml.model_loader import get_model_loader
from app.config import settings

logger = logging.getLogger(__name__)


class RiskPredictor:
    """
    Risk prediction engine
    Combines rule-based and ML-based risk scoring
    """

    def __init__(self):
        self.model_loader = get_model_loader()
        self.model = self.model_loader.get_risk_model()
        self.scaler = self.model_loader.get_risk_scaler()
        self.feature_names = self.model_loader.get_risk_features()
        self.encoders = self.model_loader.get_risk_encoders()

    def predict_risk(self, features: Dict) -> Tuple[float, str, Dict[str, float]]:
        """
        Predict risk score and level

        Args:
            features: Dictionary of engineered features

        Returns:
            Tuple of (risk_score, risk_level, risk_factors)
        """
        try:
            # Extract features in correct order
            feature_vector = self._extract_feature_vector(features)

            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)

            # Get model prediction
            risk_probabilities = self.model.predict_proba(feature_vector_scaled)[0]

            # Calculate weighted risk score (0 to 1)
            # For multi-class: weight by class probabilities
            # Low=0, Medium=1, High=2
            risk_score = (
                risk_probabilities[0] * 0.0 +  # Low risk
                risk_probabilities[1] * 0.5 +  # Medium risk
                risk_probabilities[2] * 1.0  # High risk
            )

            # Classify risk level
            risk_level = self._classify_risk_level(risk_score)

            # Calculate individual risk factors
            risk_factors = self._calculate_risk_factors(features)

            logger.info(f"Risk prediction: score={risk_score:.4f}, level={risk_level}")

            return risk_score, risk_level, risk_factors

        except Exception as e:
            logger.error(f"Risk prediction failed: {str(e)}")
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

    def _classify_risk_level(self, risk_score: float) -> str:
        """
        Classify risk score into level

        Args:
            risk_score: Numerical risk score (0-1)

        Returns:
            Risk level string (low, medium, high)
        """
        if risk_score < settings.RISK_LOW_THRESHOLD:
            return "low"
        elif risk_score < settings.RISK_HIGH_THRESHOLD:
            return "medium"
        else:
            return "high"

    def _calculate_risk_factors(self, features: Dict) -> Dict[str, float]:
        """
        Calculate individual risk factor contributions

        Args:
            features: Dictionary of features

        Returns:
            Dictionary of risk factors and their scores
        """
        risk_factors = {}

        # High claim amount
        if 'claim_amount_ratio' in features:
            claim_ratio = features['claim_amount_ratio']
            if claim_ratio > 0.7:
                risk_factors['high_claim_amount'] = 0.85
            elif claim_ratio > 0.4:
                risk_factors['high_claim_amount'] = 0.5
            else:
                risk_factors['high_claim_amount'] = 0.1

        # Multiple recent claims
        if 'claims_last_30_days' in features:
            recent_claims = features['claims_last_30_days']
            if recent_claims > 2:
                risk_factors['multiple_recent_claims'] = 0.7
            elif recent_claims > 0:
                risk_factors['multiple_recent_claims'] = 0.3
            else:
                risk_factors['multiple_recent_claims'] = 0.0

        # Rapid succession
        if features.get('is_rapid_succession_numeric', 0) == 1:
            risk_factors['rapid_succession'] = 0.8
        else:
            risk_factors['rapid_succession'] = 0.0

        # Unusual geolocation
        if features.get('is_geolocation_anomaly_numeric', 0) == 1:
            risk_factors['unusual_geolocation'] = 0.6
        else:
            risk_factors['unusual_geolocation'] = 0.0

        # New/untrusted device
        if features.get('is_untrusted_device', 0) == 1:
            risk_factors['new_device'] = 0.4
        else:
            risk_factors['new_device'] = 0.0

        # Unusual time
        if features.get('is_unusual_time_numeric', 0) == 1:
            risk_factors['unusual_time'] = 0.3
        else:
            risk_factors['unusual_time'] = 0.0

        # Suspicious session
        if features.get('is_suspicious_session', 0) == 1:
            risk_factors['suspicious_session'] = 0.5
        else:
            risk_factors['suspicious_session'] = 0.0

        return risk_factors

    def get_top_risk_factors(self, risk_factors: Dict[str, float], top_n: int = 5) -> list:
        """
        Get top N risk factors

        Args:
            risk_factors: Dictionary of risk factors
            top_n: Number of top factors to return

        Returns:
            List of tuples (factor_name, score)
        """
        sorted_factors = sorted(
            risk_factors.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_factors[:top_n]
