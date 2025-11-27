"""
Fraud Service
Handles fraud detection and analysis
"""

from sqlalchemy.orm import Session
from typing import Dict
from datetime import datetime
import datetime as dt
import logging

from app.models.claim import Claim
from app.models.user import User
from app.models.fraud_detection import FraudDetection
from app.ml.fraud_predictor import FraudPredictor
from app.ml.explainer import ExplainabilityEngine
from app.ml.model_loader import get_model_loader

logger = logging.getLogger(__name__)


class FraudService:
    """Service for fraud detection and analysis"""

    def __init__(self):
        self.fraud_predictor = FraudPredictor()
        self.explainer = ExplainabilityEngine()
        self.model_loader = get_model_loader()

    def detect_fraud(
        self,
        claim: Claim,
        user: User,
        fraud_features: Dict,
        db: Session
    ) -> FraudDetection:
        """
        Detect fraud for a claim

        Args:
            claim: Claim object
            user: User object
            fraud_features: Engineered fraud features
            db: Database session

        Returns:
            FraudDetection object
        """
        try:
            # Detect fraud
            is_suspicious, fraud_probability, predicted_fraud_type, anomaly_score, model_used = \
                self.fraud_predictor.detect_fraud(fraud_features)

            # Get confidence level
            confidence_level = self.fraud_predictor.get_confidence_level(fraud_probability)

            # Create fraud detection record (without SHAP initially)
            fraud_detection = FraudDetection(
                claim_id=claim.claim_id,
                user_id=user.user_id,
                is_suspicious=is_suspicious,
                fraud_probability=fraud_probability,
                predicted_fraud_type=predicted_fraud_type,
                anomaly_score=anomaly_score,
                model_used=model_used,
                model_version="1.0",
                detected_at=datetime.now(dt.UTC)
            )

            db.add(fraud_detection)
            db.flush()

            logger.info(
                f"Fraud detection for claim {claim.claim_number}: "
                f"suspicious={is_suspicious}, probability={fraud_probability:.4f}, "
                f"type={predicted_fraud_type}"
            )

            return fraud_detection

        except Exception as e:
            logger.error(f"Fraud detection failed: {str(e)}")
            raise

    def get_fraud_explanation(
        self,
        claim_id,
        db: Session
    ) -> Dict:
        """
        Get fraud detection with SHAP explanation

        Args:
            claim_id: Claim ID
            db: Database session

        Returns:
            Dictionary with fraud details and SHAP explanation
        """
        fraud_detection = db.query(FraudDetection).filter(
            FraudDetection.claim_id == claim_id
        ).first()

        if not fraud_detection:
            raise ValueError(f"Fraud detection not found for claim {claim_id}")

        # If SHAP values not yet calculated, calculate them now
        if not fraud_detection.shap_values:
            self._calculate_shap_explanation(fraud_detection, db)

        # Generate human-readable explanation
        explanation_text = self._generate_fraud_explanation(
            is_suspicious=fraud_detection.is_suspicious,
            fraud_probability=float(fraud_detection.fraud_probability),
            predicted_fraud_type=fraud_detection.predicted_fraud_type,
            top_features=fraud_detection.top_contributing_features or []
        )

        # Get confidence level
        confidence_level = self.fraud_predictor.get_confidence_level(
            float(fraud_detection.fraud_probability)
        )

        return {
            'detection_id': fraud_detection.detection_id,
            'claim_id': fraud_detection.claim_id,
            'is_suspicious': fraud_detection.is_suspicious,
            'fraud_probability': float(fraud_detection.fraud_probability),
            'predicted_fraud_type': fraud_detection.predicted_fraud_type,
            'shap_values': fraud_detection.shap_values,
            'top_features': fraud_detection.top_contributing_features,
            'confidence_level': confidence_level,
            'explanation': explanation_text,
            'detected_at': fraud_detection.detected_at
        }

    def _calculate_shap_explanation(
        self,
        fraud_detection: FraudDetection,
        db: Session
    ):
        """
        Calculate SHAP explanation for fraud detection

        Args:
            fraud_detection: FraudDetection object
            db: Database session
        """
        try:
            # Get claim and reconstruct features
            claim = db.query(Claim).filter(
                Claim.claim_id == fraud_detection.claim_id
            ).first()

            if not claim:
                logger.error("Claim not found for SHAP calculation")
                return

            # Get transaction context
            from app.models.transaction_context import TransactionContext

            context = db.query(TransactionContext).filter(
                TransactionContext.claim_id == claim.claim_id
            ).first()

            if not context:
                logger.error("Transaction context not found for SHAP calculation")
                return

            # Reconstruct context_data dictionary
            context_data = {
                'geolocation_distance_km': float(
                    context.geolocation_distance_km) if context.geolocation_distance_km else 0,
                'is_geolocation_anomaly': context.is_geolocation_anomaly,
                'device_trust_score': float(context.device_trust_score) if context.device_trust_score else 0.5,
                'is_trusted_device': context.is_trusted_device,
                'form_fill_time': context.form_fill_time or 300,
                'session_duration': context.session_duration or 600,
                'is_unusual_time': context.is_unusual_time,
                'transaction_hour': context.transaction_hour,
                'transaction_day_of_week': context.transaction_day_of_week,
            }

            # Re-engineer fraud features
            from app.services.feature_engineering import FeatureEngineeringService

            claim_data = {
                'claim_type': claim.claim_type,
                'claim_amount': float(claim.claim_amount),
                'supporting_documents_count': claim.supporting_documents_count
            }

            fraud_features = FeatureEngineeringService.engineer_fraud_features(
                claim_data=claim_data,
                user=claim.user,
                context_data=context_data,
                db=db
            )

            # Extract and scale feature vector
            feature_vector = self.fraud_predictor._extract_feature_vector(fraud_features)
            feature_vector_scaled = self.fraud_predictor.scaler.transform(feature_vector)

            # Generate SHAP explanation
            shap_explanation = self.explainer.explain_fraud_prediction(
                feature_vector_scaled,
                top_n=5
            )

            # Update fraud detection record
            fraud_detection.shap_values = shap_explanation['shap_values']
            fraud_detection.top_contributing_features = shap_explanation['top_features']

            db.commit()

            logger.info(f"SHAP explanation calculated for fraud detection {fraud_detection.detection_id}")

        except Exception as e:
            logger.error(f"SHAP calculation failed: {str(e)}")

    def _generate_fraud_explanation(
        self,
        is_suspicious: bool,
        fraud_probability: float,
        predicted_fraud_type: str,
        top_features: list
    ) -> str:
        """
        Generate human-readable fraud explanation

        Args:
            is_suspicious: Whether claim is suspicious
            fraud_probability: Fraud probability
            predicted_fraud_type: Predicted fraud type
            top_features: Top contributing features

        Returns:
            Explanation string
        """
        if is_suspicious:
            explanation = f"⚠️ **Fraud Alert:** This claim has been flagged as suspicious with a "
            explanation += f"{fraud_probability * 100:.1f}% fraud probability.\n\n"

            if predicted_fraud_type:
                fraud_type_name = predicted_fraud_type.replace('_', ' ').title()
                explanation += f"**Suspected Fraud Type:** {fraud_type_name}\n\n"
        else:
            explanation = f"✅ **Low Fraud Risk:** This claim appears legitimate with a "
            explanation += f"{fraud_probability * 100:.1f}% fraud probability.\n\n"

        if top_features:
            explanation += "**Key Indicators:**\n"
            for i, feature in enumerate(top_features[:3], 1):
                feature_name = feature.get('feature', 'Unknown')
                contribution = feature.get('contribution', 'unknown')

                if contribution == 'increases':
                    explanation += f"{i}. ⚠️ {feature_name}: Increases fraud suspicion\n"
                else:
                    explanation += f"{i}. ✅ {feature_name}: Decreases fraud suspicion\n"

        return explanation
