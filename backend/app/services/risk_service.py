from sqlalchemy.orm import Session
from typing import Dict
import datetime as dt
from datetime import datetime
import logging

from app.models.claim import Claim
from app.models.user import User
from app.models.risk_score import RiskScore
from app.ml.risk_predictor import RiskPredictor

logger = logging.getLogger(__name__)


class RiskService:
    """Service for risk assessment and scoring"""

    def __init__(self):
        self.risk_predictor = RiskPredictor()

    def calculate_risk_score(
        self,
        claim: Claim,
        user: User,
        risk_features: Dict,
        db: Session
    ) -> RiskScore:
        """
        Calculate risk score for a claim

        Args:
            claim: Claim object
            user: User object
            risk_features: Engineered risk features
            db: Database session

        Returns:
            RiskScore object
        """
        try:
            # Predict risk
            risk_score, risk_level, risk_factors = self.risk_predictor.predict_risk(risk_features)

            # Create risk score record
            risk_score_record = RiskScore(
                claim_id=claim.claim_id,
                user_id=user.user_id,
                risk_score=risk_score,
                risk_level=risk_level,
                factors=risk_factors,
                model_version="1.0",
                calculation_method="ml_enhanced",
                calculated_at=datetime.now(dt.UTC)
            )

            db.add(risk_score_record)
            db.flush()

            logger.info(
                f"Risk calculated for claim {claim.claim_number}: "
                f"score={risk_score:.4f}, level={risk_level}"
            )

            return risk_score_record

        except Exception as e:
            logger.error(f"Risk calculation failed: {str(e)}")
            raise

    def get_risk_explanation(
        self,
        claim_id,
        db: Session
    ) -> Dict:
        """
        Get risk score with explanation

        Args:
            claim_id: Claim ID
            db: Database session

        Returns:
            Dictionary with risk details and explanation
        """
        risk_score_record = db.query(RiskScore).filter(
            RiskScore.claim_id == claim_id
        ).first()

        if not risk_score_record:
            raise ValueError(f"Risk score not found for claim {claim_id}")

        # Get top risk factors
        top_factors = self.risk_predictor.get_top_risk_factors(
            risk_score_record.factors,
            top_n=5
        )

        # Generate explanation
        explanation = self._generate_explanation(
            risk_level=risk_score_record.risk_level,
            risk_score=float(risk_score_record.risk_score),
            top_factors=top_factors
        )

        return {
            'risk_score': float(risk_score_record.risk_score),
            'risk_level': risk_score_record.risk_level,
            'factors': risk_score_record.factors,
            'top_factors': [
                {'factor': factor, 'score': score}
                for factor, score in top_factors
            ],
            'explanation': explanation,
            'calculated_at': risk_score_record.calculated_at
        }

    def _generate_explanation(
        self,
        risk_level: str,
        risk_score: float,
        top_factors: list
    ) -> str:
        """
        Generate human-readable risk explanation

        Args:
            risk_level: Risk level (low, medium, high)
            risk_score: Numerical risk score
            top_factors: List of top risk factors

        Returns:
            Explanation string
        """
        explanation = f"This transaction has been classified as **{risk_level} risk** "
        explanation += f"with a risk score of {risk_score:.2f}.\n\n"

        if top_factors:
            explanation += "**Key Risk Factors:**\n"
            for i, (factor, score) in enumerate(top_factors[:3], 1):
                factor_name = factor.replace('_', ' ').title()
                explanation += f"{i}. {factor_name}: {score:.2f}\n"

        if risk_level == 'high':
            explanation += "\n⚠️ **Action Required:** Additional authentication is required due to high risk indicators."
        elif risk_level == 'medium':
            explanation += "\n⚡ **Moderate Risk:** Standard verification procedures will be applied."
        else:
            explanation += "\n✅ **Low Risk:** Transaction can proceed with minimal verification."

        return explanation
