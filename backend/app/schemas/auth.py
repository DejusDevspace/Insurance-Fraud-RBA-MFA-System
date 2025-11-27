from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from uuid import UUID


class Token(BaseModel):
    """JWT token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Data encoded in JWT token"""
    user_id: Optional[UUID] = None
    email: Optional[str] = None


class LoginRequest(BaseModel):
    """Login request payload"""
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    """Login response with tokens and user info"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict  # UserResponse serialized


class RegisterRequest(BaseModel):
    """Registration request payload"""
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    phone_number: Optional[str] = None

    # Address
    city: Optional[str] = None
    state: Optional[str] = None
    country: str = "USA"

    # Policy
    policy_type: Optional[str] = Field(None, pattern="^(auto|home|health|life)$")


class RegisterResponse(BaseModel):
    """Registration response"""
    message: str
    user: dict  # UserResponse serialized
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class MFARequest(BaseModel):
    """MFA verification request"""
    claim_id: UUID
    otp_code: Optional[str] = Field(None, min_length=6, max_length=6)
    biometric_verified: Optional[bool] = None


class MFAResponse(BaseModel):
    """MFA verification response"""
    success: bool
    message: str
    claim_status: Optional[str] = None
