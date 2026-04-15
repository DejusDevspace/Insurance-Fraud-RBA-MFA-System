from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
import datetime as dt
from datetime import datetime

from app.database import get_db
from app.schemas.claim import (
    ClaimCreate,
    ClaimResponse,
    ClaimListResponse,
    ClaimDetailResponse,
    ClaimSubmissionResponse
)
from app.services.claim_service import ClaimService
from app.services.mfa_service import MFAService
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter()


# TODO: Ensure features are implemented correctly
# PROBE: fingerprinting and session timing management (frontend vs backend)
@router.post("/", response_model=ClaimSubmissionResponse, status_code=status.HTTP_201_CREATED)
def submit_claim(
    claim_data: ClaimCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submit a new insurance claim

    This endpoint processes the claim through the complete pipeline:
    1. Creates claim record
    2. Captures transaction context
    3. Engineers features
    4. Calculates risk score
    5. Detects fraud
    6. Makes decision (approve/MFA/block)

    **Request Body:**
    - **claim_type**: Type of claim (accident, theft, medical, property_damage, other)
    - **claim_amount**: Claim amount in dollars
    - **incident_date**: Date of incident
    - **claim_description**: Detailed description of incident
    - **supporting_documents_count**: Number of supporting documents (optional)

    **Response includes:**
    - Claim information
    - Risk assessment
    - Whether MFA is required
    - MFA method (if required)
    """
    try:
        # Get client information
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get('user-agent', '')
        print('----- USER AGENT: ', user_agent)
        print('----- CLAIM SUB DATA: ', claim_data)

        # Simulate device fingerprint (in production, use proper fingerprinting library)
        import hashlib
        device_fingerprint = hashlib.md5(f"{user_agent}-{client_ip}".encode()).hexdigest()

        # Get transaction timing
        now = datetime.now(dt.UTC)
        transaction_hour = now.hour
        transaction_day = now.weekday()

        # Build context data. Use demo overrides from frontend if provided to simulate varying conditions
        context_data = {
            'ip_address': client_ip,
            'geolocation': {
                'city': current_user.city,
                'state': current_user.state,
                'country': current_user.country
            },
            'is_geolocation_anomaly': claim_data.is_geolocation_anomaly if claim_data.is_geolocation_anomaly is not None else False,
            'geolocation_distance_km': claim_data.geolocation_distance_km if claim_data.geolocation_distance_km is not None else 120,
            'device': {
                'fingerprint': device_fingerprint,
                'type': 'desktop',
                'os': 'Unknown',
                'browser': 'Unknown'
            },
            'device_trust_score': claim_data.device_trust_score if claim_data.device_trust_score is not None else 0.5,
            'is_trusted_device': claim_data.is_trusted_device if claim_data.is_trusted_device is not None else False,
            'session_duration': claim_data.session_duration,
            'pages_visited': claim_data.pages_visited,
            'form_fill_time': claim_data.form_fill_time,
            'is_unusual_time': False,
            'transaction_hour': transaction_hour,
            'transaction_day_of_week': transaction_day
        }

        # Submit claim through service
        claim, decision = ClaimService.submit_claim(
            user=current_user,
            claim_data=claim_data.dict(),
            context_data=context_data,
            db=db
        )

        # If MFA required and method is OTP, generate OTP
        if decision.get('requires_mfa') and decision.get('mfa_method') == 'otp':
            print('= * 30', 'SENDING OTP', '= * 30')
            otp = MFAService.generate_otp(str(claim.claim_id))
            MFAService.send_otp_notification(current_user, otp)

        # Build response
        claim_response = ClaimResponse.from_orm(claim)
        claim_response.requires_mfa = decision.get('requires_mfa')
        claim_response.mfa_method = decision.get('mfa_method')
        claim_response.risk_level = decision.get('risk_level')

        return ClaimSubmissionResponse(
            claim=claim_response,
            risk_assessment={
                'risk_score': decision.get('risk_score'),
                'risk_level': decision.get('risk_level'),
                'fraud_probability': decision.get('fraud_probability')
            },
            requires_mfa=decision.get('requires_mfa', False),
            mfa_method=decision.get('mfa_method'),
            message=decision.get('message', 'Claim submitted successfully')
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Claim submission failed: {str(e)}"
        )


@router.get("/", response_model=List[ClaimListResponse])
def get_user_claims(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0
):
    """
    Get all claims for the authenticated user

    **Query Parameters:**
    - **limit**: Maximum number of claims to return (default: 50)
    - **offset**: Number of claims to skip (default: 0)

    Returns list of claims ordered by submission date (newest first)
    """
    claims = ClaimService.get_user_claims(
        user_id=current_user.user_id,
        db=db,
        limit=limit,
        offset=offset
    )

    return [ClaimListResponse.from_orm(claim) for claim in claims]


@router.get("/{claim_id}", response_model=ClaimDetailResponse)
def get_claim_details(
    claim_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific claim

    **Path Parameters:**
    - **claim_id**: UUID of the claim

    Returns detailed claim information including risk and fraud assessments
    """
    claim = ClaimService.get_claim_by_id(claim_id, db)

    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found"
        )

    # Verify ownership (users can only see their own claims, unless admin)
    if claim.user_id != current_user.user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this claim"
        )

    # Build detailed response
    claim_detail = ClaimDetailResponse.from_orm(claim)

    # Add risk and fraud information
    if claim.risk_scores:
        latest_risk = claim.risk_scores[-1]
        claim_detail.risk_level = latest_risk.risk_level

    if claim.fraud_detections:
        latest_fraud = claim.fraud_detections[-1]
        claim_detail.fraud_probability = float(latest_fraud.fraud_probability)
        claim_detail.is_suspicious = latest_fraud.is_suspicious

    return claim_detail


@router.get("/{claim_id}/status")
def get_claim_status(
    claim_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current status of a claim

    **Path Parameters:**
    - **claim_id**: UUID of the claim

    Returns current claim status and processing information
    """
    claim = ClaimService.get_claim_by_id(claim_id, db)

    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found"
        )

    # Verify ownership
    if claim.user_id != current_user.user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this claim"
        )

    return {
        'claim_id': str(claim.claim_id),
        'claim_number': claim.claim_number,
        'claim_status': claim.claim_status,
        'approval_status': claim.approval_status,
        'submitted_at': claim.submitted_at,
        'processed_at': claim.processed_at,
        'rejection_reason': claim.rejection_reason
    }
