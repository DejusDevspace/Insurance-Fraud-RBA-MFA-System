from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime
from uuid import UUID


class RiskFactorsResponse(BaseModel):
    """Individual risk factors contributing to score"""
    high_claim_amount: Optional[float] = None
    multiple_recent_claims: Optional[float] = None
    rapid_succession: Optional[float] = None
    unusual_geolocation: Optional[float] = None
    new_device: Optional[float] = None
    unusual_time: Optional[float] = None
    suspicious_session: Optional[float] = None


class RiskScoreResponse(BaseModel):
    """Risk score details"""
    risk_score_id: UUID
    claim_id: UUID
    risk_score: float
    risk_level: str  # low, medium, high
    factors: Dict[str, float]
    model_version: Optional[str]
    calculation_method: str
    calculated_at: datetime

    class Config:
        from_attributes = True


class RiskAssessmentResponse(BaseModel):
    """Complete risk assessment response"""
    risk_score: float
    risk_level: str
    factors: Dict[str, float]
    top_risk_factors: list
    requires_mfa: bool
    mfa_method: Optional[str] = None
    explanation: str


class RiskHistoryResponse(BaseModel):
    """User's risk history"""
    user_id: UUID
    risk_assessments: list
    average_risk_score: float
    risk_trend: str  # increasing, stable, decreasing
