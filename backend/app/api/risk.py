from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from uuid import UUID

from app.database import get_db
from app.schemas.risk import RiskScoreResponse, RiskAssessmentResponse
from app.services.risk_service import RiskService
from app.services.claim_service import ClaimService
from app.api.deps import get_current_user
from app.models.user import User
from app.models.risk_score import RiskScore

router = APIRouter()


@router.get("/{claim_id}", response_model=RiskScoreResponse)
def get_risk_score(
    claim_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get risk score for a specific claim

    **Path Parameters:**
    - **claim_id**: UUID of the claim

    Returns risk score and contributing factors
    """
    # Verify claim exists and user owns it
    claim = ClaimService.get_claim_by_id(claim_id, db)

    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found"
        )

    if claim.user_id != current_user.user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this claim"
        )

    # Get risk score
    risk_score = db.query(RiskScore).filter(
        RiskScore.claim_id == claim_id
    ).first()

    if not risk_score:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Risk score not found for this claim"
        )

    return RiskScoreResponse.from_orm(risk_score)


@router.get("/{claim_id}/explanation", response_model=RiskAssessmentResponse)
def get_risk_explanation(
    claim_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed risk assessment with human-readable explanation

    **Path Parameters:**
    - **claim_id**: UUID of the claim

    Returns risk assessment with:
    - Risk score and level
    - Contributing factors
    - Top risk factors
    - MFA requirements
    - Human-readable explanation
    """
    # Verify claim exists and user owns it
    claim = ClaimService.get_claim_by_id(claim_id, db)

    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found"
        )

    if claim.user_id != current_user.user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this claim"
        )

    # Get risk explanation
    risk_service = RiskService()

    try:
        explanation = risk_service.get_risk_explanation(claim_id, db)

        # Add MFA requirements
        from app.services.decision_engine import DecisionEngine

        mfa_requirements = DecisionEngine.get_mfa_requirements(
            risk_level=explanation['risk_level'],
            fraud_probability=0.0  # Will be overridden if fraud data exists
        )

        return RiskAssessmentResponse(
            risk_score=explanation['risk_score'],
            risk_level=explanation['risk_level'],
            factors=explanation['factors'],
            top_risk_factors=explanation['top_factors'],
            requires_mfa=mfa_requirements['requires_mfa'],
            mfa_method=mfa_requirements['mfa_method'],
            explanation=explanation['explanation']
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get("/user/history")
def get_user_risk_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 10
):
    """
    Get risk assessment history for the current user

    **Query Parameters:**
    - **limit**: Maximum number of risk assessments to return (default: 10)

    Returns list of risk assessments for user's claims
    """
    risk_scores = db.query(RiskScore).filter(
        RiskScore.user_id == current_user.user_id
    ).order_by(
        RiskScore.calculated_at.desc()
    ).limit(limit).all()

    if not risk_scores:
        return {
            'user_id': str(current_user.user_id),
            'risk_assessments': [],
            'average_risk_score': 0.0,
            'risk_trend': 'stable'
        }

    # Calculate average
    avg_risk = sum(float(rs.risk_score) for rs in risk_scores) / len(risk_scores)

    # Determine trend (simple: compare first half vs second half)
    if len(risk_scores) >= 4:
        first_half_avg = sum(float(rs.risk_score) for rs in risk_scores[:len(risk_scores) // 2]) / (
                    len(risk_scores) // 2)
        second_half_avg = sum(float(rs.risk_score) for rs in risk_scores[len(risk_scores) // 2:]) / (
                    len(risk_scores) - len(risk_scores) // 2)

        if second_half_avg > first_half_avg * 1.1:
            trend = 'increasing'
        elif second_half_avg < first_half_avg * 0.9:
            trend = 'decreasing'
        else:
            trend = 'stable'
    else:
        trend = 'stable'

    return {
        'user_id': str(current_user.user_id),
        'risk_assessments': [
            {
                'risk_score_id': str(rs.risk_score_id),
                'claim_id': str(rs.claim_id),
                'risk_score': float(rs.risk_score),
                'risk_level': rs.risk_level,
                'calculated_at': rs.calculated_at
            }
            for rs in risk_scores
        ],
        'average_risk_score': avg_risk,
        'risk_trend': trend
    }
