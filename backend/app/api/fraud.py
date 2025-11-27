from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from uuid import UUID

from app.database import get_db
from app.schemas.fraud import FraudDetectionResponse, FraudExplanationResponse
from app.services.fraud_service import FraudService
from app.services.claim_service import ClaimService
from app.api.deps import get_current_user
from app.models.user import User
from app.models.fraud_detection import FraudDetection

router = APIRouter()


@router.get("/{claim_id}", response_model=FraudDetectionResponse)
def get_fraud_detection(
    claim_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get fraud detection result for a specific claim

    **Path Parameters:**
    - **claim_id**: UUID of the claim

    Returns fraud detection analysis
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

    # Get fraud detection
    fraud_detection = db.query(FraudDetection).filter(
        FraudDetection.claim_id == claim_id
    ).first()

    if not fraud_detection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fraud detection not found for this claim"
        )

    return FraudDetectionResponse.from_orm(fraud_detection)


@router.get("/{claim_id}/explanation", response_model=FraudExplanationResponse)
def get_fraud_explanation(
    claim_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed fraud detection with SHAP explanation

    **Path Parameters:**
    - **claim_id**: UUID of the claim

    Returns fraud detection with:
    - Fraud probability
    - Predicted fraud type
    - SHAP values
    - Top contributing features
    - Human-readable explanation
    - Confidence level
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

    # Get fraud explanation
    fraud_service = FraudService()

    try:
        explanation = fraud_service.get_fraud_explanation(claim_id, db)

        return FraudExplanationResponse(
            detection_id=explanation['detection_id'],
            claim_id=explanation['claim_id'],
            is_suspicious=explanation['is_suspicious'],
            fraud_probability=explanation['fraud_probability'],
            predicted_fraud_type=explanation['predicted_fraud_type'],
            shap_values=explanation['shap_values'],
            top_features=explanation['top_features'],
            base_value=explanation.get('base_value', 0.0),
            explanation=explanation['explanation'],
            confidence_level=explanation['confidence_level']
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
