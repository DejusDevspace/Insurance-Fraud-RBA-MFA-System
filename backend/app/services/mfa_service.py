from sqlalchemy.orm import Session
from typing import Dict, Optional
import datetime as dt
from datetime import datetime, timedelta
import random
import string
import logging

from app.models.claim import Claim
from app.models.user import User
from app.models.authentication_event import AuthenticationEvent
from app.config import settings
from app.core.constants import AuthEventType, MFAMethod, ClaimStatus

logger = logging.getLogger(__name__)


class MFAService:
    """Service for multi-factor authentication"""

    # In-memory OTP storage (for demo purposes)
    # In prod, it will use Redis or database
    _otp_storage: Dict[str, Dict] = {}

    @staticmethod
    def generate_otp(claim_id: str) -> str:
        """
        Generate OTP for a claim

        Args:
            claim_id: Claim ID

        Returns:
            OTP code
        """
        # Generate 6-digit OTP
        otp = ''.join(random.choices(string.digits, k=settings.OTP_LENGTH))

        # Store OTP with expiry
        expiry = datetime.now(dt.UTC) + timedelta(minutes=settings.OTP_EXPIRY_MINUTES)

        MFAService._otp_storage[claim_id] = {
            'otp': otp,
            'expiry': expiry,
            'attempts': 0
        }

        logger.info(f"OTP generated for claim {claim_id}: {otp} (expires at {expiry})")

        return otp

    @staticmethod
    def verify_otp(
        claim_id: str,
        otp_code: str,
        db: Session
    ) -> Dict:
        """
        Verify OTP for a claim

        Args:
            claim_id: Claim ID
            otp_code: OTP code to verify
            db: Database session

        Returns:
            Verification result dictionary
        """
        # Get stored OTP
        stored_data = MFAService._otp_storage.get(claim_id)

        if not stored_data:
            return {
                'success': False,
                'message': 'No OTP found. Please request a new one.',
                'reason': 'otp_not_found'
            }

        # Check expiry
        if datetime.now(dt.UTC) > stored_data['expiry']:
            del MFAService._otp_storage[claim_id]
            return {
                'success': False,
                'message': 'OTP has expired. Please request a new one.',
                'reason': 'otp_expired'
            }

        # Check attempts
        if stored_data['attempts'] >= 3:
            del MFAService._otp_storage[claim_id]
            return {
                'success': False,
                'message': 'Too many failed attempts. Please request a new OTP.',
                'reason': 'too_many_attempts'
            }

        # Verify OTP
        if otp_code == stored_data['otp']:
            # OTP correct - process claim
            del MFAService._otp_storage[claim_id]

            # Get claim
            claim = db.query(Claim).filter(Claim.claim_id == claim_id).first()

            if claim:
                # Log successful MFA
                auth_event = AuthenticationEvent(
                    user_id=claim.user_id,
                    claim_id=claim.claim_id,
                    event_type=AuthEventType.MFA_SUCCESS.value,
                    auth_method=MFAMethod.OTP.value,
                    auth_result='success',
                    mfa_required=True,
                    mfa_completed=True
                )
                db.add(auth_event)

                # Approve claim
                claim.claim_status = ClaimStatus.APPROVED.value
                claim.approved_amount = float(claim.claim_amount)
                claim.processed_at = datetime.now(dt.UTC)

                db.commit()

                logger.info(f"OTP verified successfully for claim {claim_id}")

            return {
                'success': True,
                'message': 'OTP verified successfully. Your claim has been approved.',
                'claim_status': ClaimStatus.APPROVED.value
            }
        else:
            # Increment attempts
            stored_data['attempts'] += 1

            # Log failed attempt
            claim = db.query(Claim).filter(Claim.claim_id == claim_id).first()
            if claim:
                auth_event = AuthenticationEvent(
                    user_id=claim.user_id,
                    claim_id=claim.claim_id,
                    event_type=AuthEventType.MFA_FAILURE.value,
                    auth_method=MFAMethod.OTP.value,
                    auth_result='failure',
                    mfa_required=True,
                    mfa_completed=False
                )
                db.add(auth_event)
                db.commit()

            remaining_attempts = 3 - stored_data['attempts']

            logger.warning(f"OTP verification failed for claim {claim_id}. Attempts remaining: {remaining_attempts}")

            return {
                'success': False,
                'message': f'Invalid OTP. {remaining_attempts} attempts remaining.',
                'reason': 'invalid_otp',
                'attempts_remaining': remaining_attempts
            }

    @staticmethod
    def verify_biometric(
        claim_id: str,
        user: User,
        db: Session
    ) -> Dict:
        """
        Simulate biometric verification
        In production, this would integrate with actual biometric APIs

        Args:
            claim_id: Claim ID
            user: User object
            db: Database session

        Returns:
            Verification result dictionary
        """
        # Simulate biometric verification (always succeeds in demo)
        # In prod: it would be with FaceID, TouchID, or other biometric services

        claim = db.query(Claim).filter(Claim.claim_id == claim_id).first()

        if not claim:
            return {
                'success': False,
                'message': 'Claim not found.',
                'reason': 'claim_not_found'
            }

        # Log successful biometric verification
        auth_event = AuthenticationEvent(
            user_id=user.user_id,
            claim_id=claim.claim_id,
            event_type=AuthEventType.MFA_SUCCESS.value,
            auth_method=MFAMethod.BIOMETRIC.value,
            auth_result='success',
            mfa_required=True,
            mfa_completed=True
        )
        db.add(auth_event)

        # Approve claim
        claim.claim_status = ClaimStatus.APPROVED.value
        claim.approved_amount = float(claim.claim_amount)
        claim.processed_at = datetime.now(dt.UTC)

        db.commit()

        logger.info(f"Biometric verified successfully for claim {claim_id}")

        return {
            'success': True,
            'message': 'Biometric verification successful. Your claim has been approved.',
            'claim_status': ClaimStatus.APPROVED.value
        }

    @staticmethod
    def send_otp_notification(user: User, otp: str):
        """
        Simulate sending OTP via SMS/Email
        In production: integrate with Twilio, SendGrid, etc.

        Args:
            user: User object
            otp: OTP code
        """
        # Simulate sending OTP
        logger.info(f"[SIMULATED] Sending OTP to {user.phone_number or user.email}: {otp}")

        # In prod, we could do:
        # - SMS via Twilio: client.messages.create(to=user.phone_number, body=f"Your OTP is: {otp}")
        # - Email via SendGrid: send_email(to=user.email, subject="OTP", body=f"Your OTP is: {otp}")

        print(f"\n{'=' * 60}")
        print(f"OTP NOTIFICATION (SIMULATION)")
        print(f"{'=' * 60}")
        print(f"To: {user.email}")
        print(f"Phone: {user.phone_number or 'Not provided'}")
        print(f"OTP Code: {otp}")
        print(f"Valid for: {settings.OTP_EXPIRY_MINUTES} minutes")
        print(f"{'=' * 60}\n")
