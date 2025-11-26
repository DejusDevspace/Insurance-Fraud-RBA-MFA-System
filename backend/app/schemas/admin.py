from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
from uuid import UUID


class DashboardStatsResponse(BaseModel):
    """Overall dashboard statistics"""
    total_claims: int
    pending_claims: int
    approved_claims: int
    rejected_claims: int

    fraud_detected: int
    fraud_rate: float

    high_risk_claims: int
    medium_risk_claims: int
    low_risk_claims: int

    total_claim_amount: float
    avg_claim_amount: float

    mfa_triggered: int
    mfa_success_rate: float

    active_users: int
    new_users_today: int


class RiskDistributionResponse(BaseModel):
    """Risk score distribution data"""
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int

    low_risk_percentage: float
    medium_risk_percentage: float
    high_risk_percentage: float

    average_risk_score: float


class FraudMetricsResponse(BaseModel):
    """Fraud detection metrics"""
    total_detections: int
    suspicious_claims: int
    fraud_rate: float

    fraud_by_type: Dict[str, int]
    avg_fraud_probability: float

    false_positives: Optional[int] = None
    false_negatives: Optional[int] = None
    detection_accuracy: Optional[float] = None


class UserRiskProfileResponse(BaseModel):
    """User risk profile for admin view"""
    user_id: UUID
    email: str
    full_name: str
    policy_number: str

    risk_category: str
    total_claims_count: int
    total_claims_amount: float
    fraud_flags_count: int

    recent_claims: List[dict]
    average_risk_score: float

    class Config:
        from_attributes = True


class RecentActivityResponse(BaseModel):
    """Recent system activity"""
    claim_id: UUID
    claim_number: str
    user_email: str
    claim_type: str
    claim_amount: float
    risk_level: str
    fraud_probability: Optional[float]
    submitted_at: datetime
    status: str


class AnalyticsTimeSeriesResponse(BaseModel):
    """Time series data for analytics"""
    date: datetime
    claims_count: int
    fraud_count: int
    total_amount: float
    avg_risk_score: float
