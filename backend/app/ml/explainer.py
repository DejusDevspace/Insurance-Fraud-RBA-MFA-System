import numpy as np
import shap
from typing import Dict, List
import logging

from app.ml.model_loader import get_model_loader

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    Generates human-readable explanations for ML predictions
    Uses SHAP (SHapley Additive exPlanations)
    """

    def __init__(self):
        self.model_loader = get_model_loader()
        self.risk_explainer = self.model_loader.get_risk_explainer()
        self.fraud_explainer = self.model_loader.get_fraud_explainer()
        self.risk_model = self.model_loader.get_risk_model()
        self.fraud_model = self.model_loader.get_fraud_classifier()
        self.risk_features = self.model_loader.get_risk_features()
        self.fraud_features = self.model_loader.get_fraud_features()

    def explain_risk_prediction(
            self,
            feature_vector: np.ndarray,
            top_n: int = 5
    ) -> Dict:
        """
        Generate SHAP explanation for risk prediction

        Args:
            feature_vector: Scaled feature vector
            top_n: Number of top features to return

        Returns:
            Dictionary with SHAP values and explanations
        """
        try:
            # Calculate SHAP values
            shap_values = self.risk_explainer.shap_values(feature_vector)

            # Get predicted class
            predicted_class = np.argmax(self.risk_model.predict(feature_vector)[0])

            # Handle multi-class SHAP values
            if isinstance(shap_values, list):
                shap_vals = shap_values[predicted_class][0]
                base_value = self.risk_explainer.expected_value[predicted_class]
            elif len(shap_values.shape) == 3:
                shap_vals = shap_values[0, :, predicted_class]
                base_value = self.risk_explainer.expected_value[predicted_class]
            else:
                shap_vals = shap_values[0]
                base_value = self.risk_explainer.expected_value

            # Create feature importance dictionary
            shap_dict = {
                feature: float(shap_val)
                for feature, shap_val in zip(self.risk_features, shap_vals)
            }

            # Get top features
            top_features = self._get_top_features(shap_dict, top_n)

            # Generate human-readable explanation
            explanation_text = self._generate_risk_explanation(top_features, predicted_class)

            return {
                'shap_values': shap_dict,
                'top_features': top_features,
                'base_value': float(base_value),
                'explanation': explanation_text
            }

        except Exception as e:
            logger.error(f"Risk explanation failed: {str(e)}")
            raise

    def explain_fraud_prediction(
            self,
            feature_vector: np.ndarray,
            top_n: int = 5
    ) -> Dict:
        """
        Generate SHAP explanation for fraud prediction

        Args:
            feature_vector: Scaled feature vector
            top_n: Number of top features to return

        Returns:
            Dictionary with SHAP values and explanations
        """
        try:
            # Calculate SHAP values
            shap_values = self.fraud_explainer.shap_values(feature_vector)

            # Handle binary classification
            if len(shap_values.shape) == 2:
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values

            base_value = self.fraud_explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[0] if len(base_value) > 0 else 0.0

            # Create feature importance dictionary
            shap_dict = {
                feature: float(shap_val)
                for feature, shap_val in zip(self.fraud_features, shap_vals)
            }

            # Get top features
            top_features = self._get_top_features(shap_dict, top_n)

            # Generate human-readable explanation
            explanation_text = self._generate_fraud_explanation(top_features)

            return {
                'shap_values': shap_dict,
                'top_features': top_features,
                'base_value': float(base_value),
                'explanation': explanation_text
            }

        except Exception as e:
            logger.error(f"Fraud explanation failed: {str(e)}")
            raise

    def _get_top_features(self, shap_dict: Dict[str, float], top_n: int) -> List[Dict]:
        """
        Get top N features by absolute SHAP value

        Args:
            shap_dict: Dictionary of feature SHAP values
            top_n: Number of top features

        Returns:
            List of feature dictionaries
        """
        sorted_features = sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        return [
            {
                'feature': self._format_feature_name(feature),
                'shap_value': shap_val,
                'contribution': 'increases' if shap_val > 0 else 'decreases',
                'magnitude': abs(shap_val)
            }
            for feature, shap_val in sorted_features
        ]

    def _format_feature_name(self, feature: str) -> str:
        """
        Format feature name to be human-readable

        Args:
            feature: Raw feature name

        Returns:
            Formatted feature name
        """
        # Convert snake_case to Title Case
        return feature.replace('_', ' ').title()

    def _generate_risk_explanation(self, top_features: List[Dict], risk_class: int) -> str:
        """
        Generate human-readable risk explanation

        Args:
            top_features: List of top contributing features
            risk_class: Predicted risk class (0=low, 1=medium, 2=high)

        Returns:
            Explanation string
        """
        risk_levels = ['low', 'medium', 'high']
        risk_level = risk_levels[risk_class]

        explanation = f"This transaction was classified as **{risk_level} risk** based on the following factors:\n\n"

        for i, feature in enumerate(top_features[:3], 1):
            feature_name = feature['feature']
            contribution = feature['contribution']

            explanation += f"{i}. **{feature_name}**: This factor {contribution} the risk score.\n"

        if risk_level == 'high':
            explanation += "\n⚠️ Due to high risk, additional authentication is required."
        elif risk_level == 'medium':
            explanation += "\n⚡ Moderate risk detected. Standard verification will be applied."
        else:
            explanation += "\n✅ Low risk detected. Transaction can proceed with minimal verification."

        return explanation

    def _generate_fraud_explanation(self, top_features: List[Dict]) -> str:
        """
        Generate human-readable fraud explanation

        Args:
            top_features: List of top contributing features

        Returns:
            Explanation string
        """
        explanation = "The fraud detection system analyzed this claim and identified the following indicators:\n\n"

        for i, feature in enumerate(top_features[:3], 1):
            feature_name = feature['feature']
            contribution = feature['contribution']

            if contribution == 'increases':
                explanation += f"{i}. **{feature_name}**: This is a red flag that increases fraud likelihood.\n"
            else:
                explanation += f"{i}. **{feature_name}**: This factor decreases fraud suspicion.\n"

        return explanation
