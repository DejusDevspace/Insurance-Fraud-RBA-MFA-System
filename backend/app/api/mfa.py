from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from uuid import UUID
from pydantic import BaseModel

from app.database import get_db
from app.schemas.auth import MFARequest, MFAResponse
from app.services.mfa_service import MFAService
from app.services.claim_service import ClaimService
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter()


class OTPRequestModel(BaseModel):
    """Request model for OTP generation"""
    claim_id: UUID


class OTPVerifyModel(BaseModel):
    """Request model for OTP verification"""
    claim_id: UUID
    otp_code: str


class BiometricVerifyModel(BaseModel):
    """Request model for biometric verification"""
    claim_id: UUID
    biometric_verified: bool = True


@router.post("/send-otp")
def send_otp(
    request: OTPRequestModel,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate and send OTP for claim verification

    **Request Body:**
    - **claim_id**: UUID of the claim requiring verification

    In production, this would send OTP via SMS/Email.
    For demo purposes, OTP is logged and can be retrieved from logs.
    """
    # Verify claim exists and belongs to user
    claim = ClaimService.get_claim_by_id(request.claim_id, db)

    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found"
        )

    if claim.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to verify this claim"
        )

    # Check if claim is pending
    if claim.claim_status != 'pending':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Claim is not pending verification (current status: {claim.claim_status})"
        )

    # Generate OTP
    otp = MFAService.generate_otp(str(request.claim_id))

    # Send OTP notification
    MFAService.send_otp_notification(current_user, otp)

    return {
        'success': True,
        'message': f'OTP sent to {current_user.email}',
        'claim_id': str(request.claim_id),
        'expires_in_minutes': 5,
        # For demo purposes, include OTP in response
        'otp_demo': otp
    }


@router.post("/verify-otp", response_model=MFAResponse)
def verify_otp(
    request: OTPVerifyModel,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Verify OTP for claim

    **Request Body:**
    - **claim_id**: UUID of the claim
    - **otp_code**: 6-digit OTP code

    If OTP is valid, claim will be approved automatically
    """
    # Verify claim exists and belongs to user
    claim = ClaimService.get_claim_by_id(request.claim_id, db)

    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found"
        )

    if claim.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to verify this claim"
        )

    # Verify OTP
    result = MFAService.verify_otp(
        claim_id=str(request.claim_id),
        otp_code=request.otp_code,
        db=db
    )

    if not result['success']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result['message']
        )

    return MFAResponse(
        success=True,
        message=result['message'],
        claim_status=result.get('claim_status')
    )


@router.post("/verify-biometric", response_model=MFAResponse)
def verify_biometric(
    request: BiometricVerifyModel,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Verify biometric authentication for high-risk claims

    **Request Body:**
    - **claim_id**: UUID of the claim
    - **biometric_verified**: Biometric verification status (default: true)

    In production, this would integrate with actual biometric APIs (FaceID, TouchID, etc.)
    For demo purposes, biometric verification is simulated
    """
    # Verify claim exists and belongs to user
    claim = ClaimService.get_claim_by_id(request.claim_id, db)

    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found"
        )

    if claim.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to verify this claim"
        )

    # Check if claim is pending
    if claim.claim_status != 'pending':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Claim is not pending verification (current status: {claim.claim_status})"
        )

    # Verify biometric
    result = MFAService.verify_biometric(
        claim_id=str(request.claim_id),
        user=current_user,
        db=db
    )

    if not result['success']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result['message']
        )

    return MFAResponse(
        success=True,
        message=result['message'],
        claim_status=result.get('claim_status')
    )


@router.get("/status/{claim_id}")
def get_mfa_status(
    claim_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Check MFA status for a claim

    **Path Parameters:**
    - **claim_id**: UUID of the claim

    Returns whether MFA is required and which method
    """
    # Verify claim exists and belongs to user
    claim = ClaimService.get_claim_by_id(claim_id, db)

    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found"
        )

    if claim.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this claim"
        )

    # Get risk and fraud data to determine MFA requirements
    from app.models.risk_score import RiskScore
    from app.models.fraud_detection import FraudDetection
    from app.services.decision_engine import DecisionEngine

    risk_score = db.query(RiskScore).filter(
        RiskScore.claim_id == claim_id
    ).first()

    fraud_detection = db.query(FraudDetection).filter(
        FraudDetection.claim_id == claim_id
    ).first()

    if not risk_score or not fraud_detection:
        return {
            'claim_id': str(claim_id),
            'requires_mfa': False,
            'mfa_method': None,
            'claim_status': claim.claim_status
        }

    # Get MFA requirements
    mfa_requirements = DecisionEngine.get_mfa_requirements(
        risk_level=risk_score.risk_level,
        fraud_probability=float(fraud_detection.fraud_probability)
    )

    return {
        'claim_id': str(claim_id),
        'requires_mfa': mfa_requirements['requires_mfa'],
        'mfa_method': mfa_requirements['mfa_method'],
        'reason': mfa_requirements['reason'],
        'claim_status': claim.claim_status,
        'risk_level': risk_score.risk_level,
        'fraud_probability': float(fraud_detection.fraud_probability)
    }
