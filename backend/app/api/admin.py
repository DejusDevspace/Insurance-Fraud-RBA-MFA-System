from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from typing import List, Optional
from datetime import datetime, timedelta
from uuid import UUID

from app.database import get_db
from app.schemas.admin import (
    DashboardStatsResponse,
    RiskDistributionResponse,
    FraudMetricsResponse,
    UserRiskProfileResponse,
    RecentActivityResponse
)
from app.schemas.fraud import FraudAlertResponse
from app.api.deps import get_current_admin_user
from app.models.user import User
from app.models.claim import Claim
from app.models.risk_score import RiskScore
from app.models.fraud_detection import FraudDetection
from app.models.authentication_event import AuthenticationEvent

router = APIRouter()


@router.get("/dashboard", response_model=DashboardStatsResponse)
def get_dashboard_stats(
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get overall dashboard statistics

    Requires admin privileges

    Returns comprehensive statistics including:
    - Total claims and status breakdown
    - Fraud detection metrics
    - Risk distribution
    - MFA statistics
    - User metrics
    """
    # Total claims
    total_claims = db.query(func.count(Claim.claim_id)).scalar()

    # Claims by status
    pending_claims = db.query(func.count(Claim.claim_id)).filter(
        Claim.claim_status == 'pending'
    ).scalar()

    approved_claims = db.query(func.count(Claim.claim_id)).filter(
        Claim.claim_status == 'approved'
    ).scalar()

    rejected_claims = db.query(func.count(Claim.claim_id)).filter(
        Claim.claim_status == 'rejected'
    ).scalar()

    # Fraud metrics
    fraud_detected = db.query(func.count(FraudDetection.detection_id)).filter(
        FraudDetection.is_suspicious == True
    ).scalar()

    fraud_rate = (fraud_detected / total_claims * 100) if total_claims > 0 else 0

    # Risk distribution
    high_risk = db.query(func.count(RiskScore.risk_score_id)).filter(
        RiskScore.risk_level == 'high'
    ).scalar()

    medium_risk = db.query(func.count(RiskScore.risk_score_id)).filter(
        RiskScore.risk_level == 'medium'
    ).scalar()

    low_risk = db.query(func.count(RiskScore.risk_score_id)).filter(
        RiskScore.risk_level == 'low'
    ).scalar()

    # Claim amounts
    total_amount = db.query(func.sum(Claim.claim_amount)).scalar() or 0
    avg_amount = db.query(func.avg(Claim.claim_amount)).scalar() or 0

    # MFA metrics
    mfa_triggered = db.query(func.count(AuthenticationEvent.event_id)).filter(
        AuthenticationEvent.event_type == 'mfa_trigger'
    ).scalar()

    mfa_success = db.query(func.count(AuthenticationEvent.event_id)).filter(
        AuthenticationEvent.event_type == 'mfa_success'
    ).scalar()

    mfa_success_rate = (mfa_success / mfa_triggered * 100) if mfa_triggered > 0 else 0

    # User metrics
    active_users = db.query(func.count(User.user_id)).filter(
        User.account_status == 'active'
    ).scalar()

    today = datetime.utcnow().date()
    new_users_today = db.query(func.count(User.user_id)).filter(
        func.date(User.account_created_at) == today
    ).scalar()

    return DashboardStatsResponse(
        total_claims=total_claims,
        pending_claims=pending_claims,
        approved_claims=approved_claims,
        rejected_claims=rejected_claims,
        fraud_detected=fraud_detected,
        fraud_rate=fraud_rate,
        high_risk_claims=high_risk,
        medium_risk_claims=medium_risk,
        low_risk_claims=low_risk,
        total_claim_amount=float(total_amount),
        avg_claim_amount=float(avg_amount),
        mfa_triggered=mfa_triggered,
        mfa_success_rate=mfa_success_rate,
        active_users=active_users,
        new_users_today=new_users_today
    )


@router.get("/risk-distribution", response_model=RiskDistributionResponse)
def get_risk_distribution(
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get risk score distribution

    Requires admin privileges

    Returns detailed breakdown of risk levels across all claims
    """
    # Count by risk level
    low_risk = db.query(func.count(RiskScore.risk_score_id)).filter(
        RiskScore.risk_level == 'low'
    ).scalar()

    medium_risk = db.query(func.count(RiskScore.risk_score_id)).filter(
        RiskScore.risk_level == 'medium'
    ).scalar()

    high_risk = db.query(func.count(RiskScore.risk_score_id)).filter(
        RiskScore.risk_level == 'high'
    ).scalar()

    total = low_risk + medium_risk + high_risk

    # Calculate percentages
    low_pct = (low_risk / total * 100) if total > 0 else 0
    medium_pct = (medium_risk / total * 100) if total > 0 else 0
    high_pct = (high_risk / total * 100) if total > 0 else 0

    # Average risk score
    avg_risk = db.query(func.avg(RiskScore.risk_score)).scalar() or 0

    return RiskDistributionResponse(
        low_risk_count=low_risk,
        medium_risk_count=medium_risk,
        high_risk_count=high_risk,
        low_risk_percentage=low_pct,
        medium_risk_percentage=medium_pct,
        high_risk_percentage=high_pct,
        average_risk_score=float(avg_risk)
    )


@router.get("/fraud-metrics", response_model=FraudMetricsResponse)
def get_fraud_metrics(
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get fraud detection metrics

    Requires admin privileges

    Returns comprehensive fraud detection statistics
    """
    # Total detections
    total_detections = db.query(func.count(FraudDetection.detection_id)).scalar()

    # Suspicious claims
    suspicious_claims = db.query(func.count(FraudDetection.detection_id)).filter(
        FraudDetection.is_suspicious == True
    ).scalar()

    # Fraud rate
    fraud_rate = (suspicious_claims / total_detections * 100) if total_detections > 0 else 0

    # Fraud by type
    fraud_types = db.query(
        FraudDetection.predicted_fraud_type,
        func.count(FraudDetection.detection_id)
    ).filter(
        FraudDetection.is_suspicious == True
    ).group_by(
        FraudDetection.predicted_fraud_type
    ).all()

    fraud_by_type = {fraud_type: count for fraud_type, count in fraud_types if fraud_type}

    # Average fraud probability
    avg_fraud_prob = db.query(func.avg(FraudDetection.fraud_probability)).scalar() or 0

    return FraudMetricsResponse(
        total_detections=total_detections,
        suspicious_claims=suspicious_claims,
        fraud_rate=fraud_rate,
        fraud_by_type=fraud_by_type,
        avg_fraud_probability=float(avg_fraud_prob)
    )


@router.get("/fraud-alerts", response_model=List[FraudAlertResponse])
def get_fraud_alerts(
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
    limit: int = 20
):
    """
    Get recent fraud alerts

    Requires admin privileges

    **Query Parameters:**
    - **limit**: Maximum number of alerts to return (default: 20)

    Returns list of suspicious claims flagged by fraud detection system
    """
    # Get recent suspicious detections
    alerts = db.query(FraudDetection).filter(
        FraudDetection.is_suspicious == True
    ).order_by(
        FraudDetection.detected_at.desc()
    ).limit(limit).all()

    # Build response
    alert_responses = []
    for alert in alerts:
        claim = db.query(Claim).filter(Claim.claim_id == alert.claim_id).first()
        user = db.query(User).filter(User.user_id == alert.user_id).first()

        if claim and user:
            alert_responses.append(
                FraudAlertResponse(
                    detection_id=alert.detection_id,
                    claim_id=alert.claim_id,
                    claim_number=claim.claim_number,
                    user_id=user.user_id,
                    user_email=user.email,
                    fraud_probability=float(alert.fraud_probability),
                    predicted_fraud_type=alert.predicted_fraud_type,
                    claim_amount=float(claim.claim_amount),
                    detected_at=alert.detected_at
                )
            )

    return alert_responses


@router.get("/claims")
def get_all_claims(
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
    status: Optional[str] = None,
    risk_level: Optional[str] = None,
    is_suspicious: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    Get all claims with filters

    Requires admin privileges

    **Query Parameters:**
    - **status**: Filter by claim status (pending, approved, rejected, under_review)
    - **risk_level**: Filter by risk level (low, medium, high)
    - **is_suspicious**: Filter by fraud suspicion (true/false)
    - **limit**: Maximum number of claims to return (default: 50)
    - **offset**: Number of claims to skip (default: 0)

    Returns paginated list of claims with filters
    """
    query = db.query(Claim)

    # Apply filters
    if status:
        query = query.filter(Claim.claim_status == status)

    if risk_level or is_suspicious is not None:
        query = query.join(RiskScore, Claim.claim_id == RiskScore.claim_id)
        if risk_level:
            query = query.filter(RiskScore.risk_level == risk_level)

    if is_suspicious is not None:
        query = query.join(FraudDetection, Claim.claim_id == FraudDetection.claim_id)
        query = query.filter(FraudDetection.is_suspicious == is_suspicious)

    # Get total count
    total = query.count()

    # Apply pagination
    claims = query.order_by(Claim.submitted_at.desc()).limit(limit).offset(offset).all()

    # Build response
    claims_data = []
    for claim in claims:
        # Get risk and fraud data
        risk_score = db.query(RiskScore).filter(RiskScore.claim_id == claim.claim_id).first()
        fraud_detection = db.query(FraudDetection).filter(FraudDetection.claim_id == claim.claim_id).first()

        claim_data = {
            'claim_id': str(claim.claim_id),
            'claim_number': claim.claim_number,
            'user_email': claim.user.email,
            'claim_type': claim.claim_type,
            'claim_amount': float(claim.claim_amount),
            'claim_status': claim.claim_status,
            'submitted_at': claim.submitted_at,
            'risk_level': risk_score.risk_level if risk_score else None,
            'risk_score': float(risk_score.risk_score) if risk_score else None,
            'is_suspicious': fraud_detection.is_suspicious if fraud_detection else None,
            'fraud_probability': float(fraud_detection.fraud_probability) if fraud_detection else None
        }
        claims_data.append(claim_data)

    return {
        'total': total,
        'limit': limit,
        'offset': offset,
        'claims': claims_data
    }


@router.get("/users/{user_id}/profile", response_model=UserRiskProfileResponse)
def get_user_risk_profile(
    user_id: UUID,
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed risk profile for a specific user

    Requires admin privileges

    **Path Parameters:**
    - **user_id**: UUID of the user

    Returns comprehensive user risk profile including claim history and risk metrics
    """
    # Get user
    user = db.query(User).filter(User.user_id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Get recent claims
    recent_claims = db.query(Claim).filter(
        Claim.user_id == user_id
    ).order_by(
        Claim.submitted_at.desc()
    ).limit(10).all()

    recent_claims_data = [
        {
            'claim_id': str(claim.claim_id),
            'claim_number': claim.claim_number,
            'claim_type': claim.claim_type,
            'claim_amount': float(claim.claim_amount),
            'claim_status': claim.claim_status,
            'submitted_at': claim.submitted_at
        }
        for claim in recent_claims
    ]

    # Calculate average risk score
    avg_risk = db.query(func.avg(RiskScore.risk_score)).filter(
        RiskScore.user_id == user_id
    ).scalar() or 0

    return UserRiskProfileResponse(
        user_id=user.user_id,
        email=user.email,
        full_name=user.full_name,
        policy_number=user.policy_number,
        risk_category=user.risk_category,
        total_claims_count=user.total_claims_count,
        total_claims_amount=float(user.total_claims_amount),
        fraud_flags_count=user.fraud_flags_count,
        recent_claims=recent_claims_data,
        average_risk_score=float(avg_risk)
    )


@router.get("/activity/recent", response_model=List[RecentActivityResponse])
def get_recent_activity(
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
    limit: int = 20
):
    """
    Get recent system activity

    Requires admin privileges

    **Query Parameters:**
    - **limit**: Maximum number of activities to return (default: 20)

    Returns recent claims with risk and fraud assessments
    """
    # Get recent claims
    claims = db.query(Claim).order_by(
        Claim.submitted_at.desc()
    ).limit(limit).all()

    activities = []
    for claim in claims:
        risk_score = db.query(RiskScore).filter(RiskScore.claim_id == claim.claim_id).first()
        fraud_detection = db.query(FraudDetection).filter(FraudDetection.claim_id == claim.claim_id).first()

        activities.append(
            RecentActivityResponse(
                claim_id=claim.claim_id,
                claim_number=claim.claim_number,
                user_email=claim.user.email,
                claim_type=claim.claim_type,
                claim_amount=float(claim.claim_amount),
                risk_level=risk_score.risk_level if risk_score else 'unknown',
                fraud_probability=float(fraud_detection.fraud_probability) if fraud_detection else None,
                submitted_at=claim.submitted_at,
                status=claim.claim_status
            )
        )

    return activities
