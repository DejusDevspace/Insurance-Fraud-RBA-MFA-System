from pydantic import BaseModel, Field, validator, field_validator
from typing import Optional, List
from datetime import date, datetime
from uuid import UUID
from decimal import Decimal


class ClaimBase(BaseModel):
    """Base claim schema"""
    claim_type: str = Field(..., regex="^(accident|theft|medical|property_damage|other)$")
    claim_amount: Decimal = Field(..., gt=0, le=1_000_000)
    incident_date: date
    claim_description: str = Field(..., min_length=10, max_length=2000)
    supporting_documents_count: int = Field(default=0, ge=0, le=10)

    @field_validator('incident_date')
    def incident_date_not_future(cls, v):
        """Ensure incident date is not in the future"""
        if v > date.today():
            raise ValueError('Incident date cannot be in the future')
        return v

    @field_validator('incident_date')
    def incident_date_not_too_old(cls, v):
        """Ensure incident date is not more than 2 years old"""
        from datetime import timedelta
        max_age = date.today() - timedelta(days=730)  # 2 years
        if v < max_age:
            raise ValueError('Incident date cannot be more than 2 years ago')
        return v


class ClaimCreate(ClaimBase):
    """Schema for creating a new claim"""
    pass


class ClaimResponse(BaseModel):
    """Basic claim response"""
    claim_id: UUID
    claim_number: str
    claim_type: str
    claim_amount: float
    incident_date: date
    claim_status: str
    submitted_at: datetime

    # Risk and fraud indicators
    requires_mfa: Optional[bool] = None
    mfa_method: Optional[str] = None
    risk_level: Optional[str] = None

    class Config:
        from_attributes = True


class ClaimListResponse(BaseModel):
    """Schema for listing claims"""
    claim_id: UUID
    claim_number: str
    claim_type: str
    claim_amount: float
    incident_date: date
    claim_status: str
    submitted_at: datetime
    processed_at: Optional[datetime]

    class Config:
        from_attributes = True


class ClaimDetailResponse(BaseModel):
    """Detailed claim response with all information"""
    claim_id: UUID
    claim_number: str
    claim_type: str
    claim_amount: float
    incident_date: date
    claim_description: str
    supporting_documents_count: int

    # Status
    claim_status: str
    submitted_at: datetime
    processed_at: Optional[datetime]

    # Decision
    approval_status: Optional[str]
    rejection_reason: Optional[str]
    approved_amount: Optional[float]

    # Risk and fraud
    risk_level: Optional[str]
    fraud_probability: Optional[float]
    is_suspicious: Optional[bool]

    class Config:
        from_attributes = True


class ClaimUpdate(BaseModel):
    """Schema for updating claim (admin only)"""
    claim_status: Optional[str] = Field(None, regex="^(pending|approved|rejected|under_review)$")
    approval_status: Optional[str] = Field(None, regex="^(approved|rejected|flagged)$")
    rejection_reason: Optional[str] = None
    approved_amount: Optional[Decimal] = None


class ClaimSubmissionResponse(BaseModel):
    """Response after claim submission with risk assessment"""
    claim: ClaimResponse
    risk_assessment: dict
    requires_mfa: bool
    mfa_method: Optional[str] = None
    message: str
