"""
Decision Engine
Combines risk and fraud assessments to make final decisions
"""

from sqlalchemy.orm import Session
from typing import Dict
import logging

from app.models.claim import Claim
from app.models.risk_score import RiskScore
from app.models.fraud_detection import FraudDetection
from app.config import settings
from app.core.constants import MFAMethod

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Decision engine for claim processing
    Combines risk score and fraud probability to determine action
    """

    @staticmethod
    def make_decision(
        claim: Claim,
        risk_score_record: RiskScore,
        fraud_detection_record: FraudDetection,
        db: Session
    ) -> Dict:
        """
        Make decision on claim based on risk and fraud analysis

        Decision Rules:
        - HIGH FRAUD (>0.8): Block immediately
        - HIGH RISK + HIGH FRAUD (>0.7): Block and flag for review
        - HIGH RISK OR MEDIUM FRAUD: Require biometric MFA
        - MEDIUM RISK OR LOW FRAUD: Require OTP
        - LOW RISK + LOW FRAUD: Approve

        Args:
            claim: Claim object
            risk_score_record: RiskScore object
            fraud_detection_record: FraudDetection object
            db: Database session

        Returns:
            Decision dictionary with action, requires_mfa, mfa_method, reason
        """
        risk_score = float(risk_score_record.risk_score)
        risk_level = risk_score_record.risk_level
        fraud_probability = float(fraud_detection_record.fraud_probability)
        is_suspicious = fraud_detection_record.is_suspicious

        logger.info(
            f"Making decision for claim {claim.claim_number}: "
            f"risk={risk_level}({risk_score:.2f}), fraud_prob={fraud_probability:.2f}"
        )

        # Decision logic
        decision = {
            'claim_id': str(claim.claim_id),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'fraud_probability': fraud_probability,
            'is_suspicious': is_suspicious
        }

        # Rule 1: Very high fraud probability - Block immediately
        if fraud_probability > 0.85:
            decision.update({
                'action': 'block',
                'requires_mfa': False,
                'mfa_method': None,
                'reason': 'Extremely high fraud probability detected',
                'message': 'This claim has been blocked due to suspected fraudulent activity.'
            })
            logger.warning(f"Claim {claim.claim_number} BLOCKED: fraud_prob={fraud_probability:.2f}")
            return decision

        # Rule 2: High risk + high fraud - Block and flag for manual review
        if risk_level == 'high' and fraud_probability > 0.7:
            decision.update({
                'action': 'block',
                'requires_mfa': False,
                'mfa_method': None,
                'reason': 'High risk combined with high fraud probability',
                'message': 'This claim requires manual review due to multiple risk indicators.'
            })
            logger.warning(f"Claim {claim.claim_number} BLOCKED: high risk + high fraud")
            return decision

        # Rule 3: High risk OR medium-high fraud - Require biometric MFA
        if risk_level == 'high' or fraud_probability > 0.6:
            decision.update({
                'action': 'require_mfa',
                'requires_mfa': True,
                'mfa_method': MFAMethod.BIOMETRIC.value,
                'reason': 'High risk or elevated fraud probability',
                'message': 'Biometric verification is required to proceed with this claim.'
            })
            logger.info(f"Claim {claim.claim_number} requires BIOMETRIC MFA")
            return decision

        # Rule 4: Medium risk OR low-medium fraud - Require OTP
        if risk_level == 'medium' or fraud_probability > 0.3:
            decision.update({
                'action': 'require_mfa',
                'requires_mfa': True,
                'mfa_method': MFAMethod.OTP.value,
                'reason': 'Moderate risk detected',
                'message': 'Please verify your identity with the OTP sent to your registered phone.'
            })
            logger.info(f"Claim {claim.claim_number} requires OTP MFA")
            return decision

        # Rule 5: Low risk + low fraud - Approve
        decision.update({
            'action': 'approve',
            'requires_mfa': False,
            'mfa_method': None,
            'reason': 'Low risk and low fraud probability',
            'message': 'Your claim has been approved and will be processed shortly.'
        })
        logger.info(f"Claim {claim.claim_number} APPROVED: low risk")
        return decision

    @staticmethod
    def get_mfa_requirements(
        risk_level: str,
        fraud_probability: float
    ) -> Dict:
        """
        Determine MFA requirements based on risk and fraud

        Args:
            risk_level: Risk level (low, medium, high)
            fraud_probability: Fraud probability (0-1)

        Returns:
            Dictionary with MFA requirements
        """
        if risk_level == 'high' or fraud_probability > 0.6:
            return {
                'requires_mfa': True,
                'mfa_method': MFAMethod.BIOMETRIC.value,
                'reason': 'High security verification required'
            }
        elif risk_level == 'medium' or fraud_probability > 0.3:
            return {
                'requires_mfa': True,
                'mfa_method': MFAMethod.OTP.value,
                'reason': 'Standard verification required'
            }
        else:
            return {
                'requires_mfa': False,
                'mfa_method': None,
                'reason': 'Low risk transaction'
            }
