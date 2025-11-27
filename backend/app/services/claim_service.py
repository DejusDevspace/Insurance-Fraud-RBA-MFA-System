from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Tuple
import datetime as dt
from datetime import datetime
import uuid
import logging

from app.models.claim import Claim
from app.models.user import User
from app.models.transaction_context import TransactionContext
from app.models.device import Device
from app.services.feature_engineering import FeatureEngineeringService
from app.services.risk_service import RiskService
from app.services.fraud_service import FraudService
from app.services.decision_engine import DecisionEngine
from app.core.constants import ClaimStatus

logger = logging.getLogger(__name__)


class ClaimService:
    """Service for managing insurance claims"""

    @staticmethod
    def submit_claim(
        user: User,
        claim_data: Dict,
        context_data: Dict,
        db: Session
    ) -> Tuple[Claim, Dict]:
        """
        Submit a new insurance claim and process it through the pipeline

        Pipeline:
        1. Create claim record
        2. Capture transaction context
        3. Engineer features
        4. Calculate risk score
        5. Detect fraud
        6. Make decision (approve/MFA/block)

        Args:
            user: User submitting claim
            claim_data: Claim information
            context_data: Transaction context (device, location, etc.)
            db: Database session

        Returns:
            Tuple of (claim, decision_result)
        """
        try:
            # Generate claim number
            claim_number = f"CLM-{uuid.uuid4().hex[:12].upper()}"

            # Create claim
            claim = Claim(
                user_id=user.user_id,
                claim_number=claim_number,
                claim_type=claim_data['claim_type'],
                claim_amount=claim_data['claim_amount'],
                incident_date=claim_data['incident_date'],
                claim_description=claim_data['claim_description'],
                supporting_documents_count=claim_data.get('supporting_documents_count', 0),
                claim_status=ClaimStatus.PENDING.value,
                submitted_at=datetime.now(dt.UTC)
            )

            db.add(claim)
            db.flush()  # Get claim_id without committing

            logger.info(f"Claim created: {claim_number} for user {user.email}")

            # Capture transaction context
            transaction_context = ClaimService._capture_transaction_context(
                claim=claim,
                user=user,
                context_data=context_data,
                db=db
            )

            db.add(transaction_context)
            db.flush()

            # Engineer features for ML models
            risk_features = FeatureEngineeringService.engineer_risk_features(
                claim_data=claim_data,
                user=user,
                context_data=context_data,
                db=db
            )

            fraud_features = FeatureEngineeringService.engineer_fraud_features(
                claim_data=claim_data,
                user=user,
                context_data=context_data,
                db=db
            )

            # Calculate risk score
            risk_service = RiskService()
            risk_score_record = risk_service.calculate_risk_score(
                claim=claim,
                user=user,
                risk_features=risk_features,
                db=db
            )

            # Detect fraud
            fraud_service = FraudService()
            fraud_detection_record = fraud_service.detect_fraud(
                claim=claim,
                user=user,
                fraud_features=fraud_features,
                db=db
            )

            # Make decision
            decision_engine = DecisionEngine()
            decision = decision_engine.make_decision(
                claim=claim,
                risk_score_record=risk_score_record,
                fraud_detection_record=fraud_detection_record,
                db=db
            )

            # Update claim status based on decision
            if decision['action'] == 'block':
                claim.claim_status = ClaimStatus.REJECTED.value
                claim.rejection_reason = decision.get('reason', 'High fraud risk detected')
            elif decision['action'] == 'approve':
                claim.claim_status = ClaimStatus.APPROVED.value
                claim.approved_amount = float(claim.claim_amount)
            # else: remains PENDING (requires MFA)

            # Commit all changes
            db.commit()
            db.refresh(claim)

            logger.info(
                f"Claim {claim_number} processed: "
                f"risk={risk_score_record.risk_level}, "
                f"fraud_prob={fraud_detection_record.fraud_probability:.2f}, "
                f"action={decision['action']}"
            )

            return claim, decision

        except Exception as e:
            db.rollback()
            logger.error(f"Claim submission failed: {str(e)}")
            raise

    @staticmethod
    def _capture_transaction_context(
        claim: Claim,
        user: User,
        context_data: Dict,
        db: Session
    ) -> TransactionContext:
        """
        Capture transaction context for the claim

        Args:
            claim: Claim object
            user: User object
            context_data: Context information
            db: Database session

        Returns:
            TransactionContext object
        """
        # Get or create device
        device = ClaimService._get_or_create_device(
            user=user,
            device_data=context_data.get('device', {}),
            db=db
        )

        # Calculate transaction timing
        now = datetime.now(dt.UTC)
        transaction_hour = now.hour
        transaction_day = now.weekday()
        is_unusual_time = (transaction_hour < 6 or transaction_hour > 22)

        # Get user's recent claims for pattern analysis
        recent_claims = db.query(Claim).filter(
            Claim.user_id == user.user_id,
            Claim.claim_id != claim.claim_id
        ).order_by(Claim.submitted_at.desc()).all()

        # Calculate days since last claim
        days_since_last = None
        if recent_claims:
            last_claim = recent_claims[0]
            days_since_last = (now - last_claim.submitted_at).days

        # Count recent claims
        from datetime import timedelta
        thirty_days_ago = now - timedelta(days=30)
        ninety_days_ago = now - timedelta(days=90)

        claims_last_30 = len([c for c in recent_claims if c.submitted_at >= thirty_days_ago])
        claims_last_90 = len([c for c in recent_claims if c.submitted_at >= ninety_days_ago])

        # Create transaction context
        transaction_context = TransactionContext(
            claim_id=claim.claim_id,
            user_id=user.user_id,
            device_id=device.device_id if device else None,
            ip_address=context_data.get('ip_address'),
            geolocation=context_data.get('geolocation'),
            is_geolocation_anomaly=context_data.get('is_geolocation_anomaly', False),
            geolocation_distance_km=context_data.get('geolocation_distance_km', 0),
            session_duration=context_data.get('session_duration', 600),
            pages_visited=context_data.get('pages_visited', 5),
            form_fill_time=context_data.get('form_fill_time', 300),
            transaction_hour=transaction_hour,
            transaction_day_of_week=transaction_day,
            is_unusual_time=is_unusual_time,
            days_since_last_claim=days_since_last,
            is_rapid_succession=(days_since_last is not None and days_since_last < 7),
            claims_last_30_days=claims_last_30,
            claims_last_90_days=claims_last_90,
            device_type=context_data.get('device', {}).get('type'),
            device_fingerprint=context_data.get('device', {}).get('fingerprint'),
            is_trusted_device=device.is_trusted if device else False,
            device_trust_score=device.device_trust_score if device else 0.5,
            captured_at=now
        )

        return transaction_context

    @staticmethod
    def _get_or_create_device(
        user: User,
        device_data: Dict,
        db: Session
    ) -> Optional[Device]:
        """
        Get existing device or create new one

        Args:
            user: User object
            device_data: Device information
            db: Database session

        Returns:
            Device object or None
        """
        device_fingerprint = device_data.get('fingerprint')

        if not device_fingerprint:
            return None

        # Check if device exists
        device = db.query(Device).filter(
            Device.device_fingerprint == device_fingerprint
        ).first()

        if device:
            # Update usage
            device.update_usage()
            return device

        # Create new device
        device = Device(
            user_id=user.user_id,
            device_fingerprint=device_fingerprint,
            device_type=device_data.get('type', 'desktop'),
            os=device_data.get('os'),
            browser=device_data.get('browser'),
            is_trusted=False,
            device_trust_score=0.3  # New devices start with low trust
        )

        db.add(device)

        return device

    @staticmethod
    def get_user_claims(
        user_id: uuid.UUID,
        db: Session,
        limit: int = 50,
        offset: int = 0
    ) -> List[Claim]:
        """
        Get claims for a specific user

        Args:
            user_id: User ID
            db: Database session
            limit: Maximum number of claims to return
            offset: Number of claims to skip

        Returns:
            List of Claim objects
        """
        claims = db.query(Claim).filter(
            Claim.user_id == user_id
        ).order_by(
            Claim.submitted_at.desc()
        ).limit(limit).offset(offset).all()

        return claims

    @staticmethod
    def get_claim_by_id(
        claim_id: uuid.UUID,
        db: Session
    ) -> Optional[Claim]:
        """
        Get claim by ID

        Args:
            claim_id: Claim ID
            db: Database session

        Returns:
            Claim object or None
        """
        return db.query(Claim).filter(Claim.claim_id == claim_id).first()

    @staticmethod
    def update_claim_status(
        claim_id: uuid.UUID,
        status: str,
        db: Session,
        approval_status: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        approved_amount: Optional[float] = None
    ) -> Claim:
        """
        Update claim status (typically after MFA or admin review)

        Args:
            claim_id: Claim ID
            status: New status
            db: Database session
            approval_status: Approval status
            rejection_reason: Reason for rejection
            approved_amount: Approved amount

        Returns:
            Updated Claim object
        """
        claim = ClaimService.get_claim_by_id(claim_id, db)

        if not claim:
            raise ValueError(f"Claim not found: {claim_id}")

        claim.claim_status = status

        if approval_status:
            claim.approval_status = approval_status

        if rejection_reason:
            claim.rejection_reason = rejection_reason

        if approved_amount is not None:
            claim.approved_amount = approved_amount

        if status in [ClaimStatus.APPROVED.value, ClaimStatus.REJECTED.value]:
            claim.processed_at = datetime.now(dt.UTC)

        db.commit()
        db.refresh(claim)

        logger.info(f"Claim {claim.claim_number} status updated to {status}")

        return claim
