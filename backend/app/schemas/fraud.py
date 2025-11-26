from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
from uuid import UUID


class ShapFeature(BaseModel):
    """SHAP feature contribution"""
    feature: str
    shap_value: float
    contribution: str  # increases, decreases
    magnitude: float


class FraudDetectionResponse(BaseModel):
    """Fraud detection result"""
    detection_id: UUID
    claim_id: UUID
    is_suspicious: bool
    fraud_probability: float
    predicted_fraud_type: Optional[str]
    anomaly_score: Optional[float]
    model_used: str
    detected_at: datetime

    class Config:
        from_attributes = True


class FraudExplanationResponse(BaseModel):
    """Fraud detection with SHAP explanation"""
    detection_id: UUID
    claim_id: UUID
    is_suspicious: bool
    fraud_probability: float
    predicted_fraud_type: Optional[str]

    # Explainability
    shap_values: Dict[str, float]
    top_features: List[ShapFeature]
    base_value: float

    # Human-readable explanation
    explanation: str
    confidence_level: str  # low, medium, high


class FraudAlertResponse(BaseModel):
    """Fraud alert for admin dashboard"""
    detection_id: UUID
    claim_id: UUID
    claim_number: str
    user_id: UUID
    user_email: str
    fraud_probability: float
    predicted_fraud_type: Optional[str]
    claim_amount: float
    detected_at: datetime

    class Config:
        from_attributes = True
