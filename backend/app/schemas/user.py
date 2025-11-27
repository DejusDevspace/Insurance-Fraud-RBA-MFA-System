from pydantic import BaseModel, EmailStr, Field, validator, field_validator
from typing import Optional
from datetime import date, datetime
from uuid import UUID


class UserBase(BaseModel):
    """Base user schema with common fields"""
    email: EmailStr
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    phone_number: Optional[str] = Field(None, max_length=20)
    date_of_birth: Optional[date] = None


class UserCreate(UserBase):
    """Schema for user registration"""
    password: str = Field(..., min_length=8, max_length=100)

    # Address
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: str = "USA"
    postal_code: Optional[str] = None

    # Policy information (optional during registration)
    policy_type: Optional[str] = Field(None, pattern="^(auto|home|health|life)$")

    @field_validator('password')
    def password_strength(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        return v


class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Schema for user data in responses"""
    user_id: UUID
    email: EmailStr
    first_name: str
    last_name: str
    phone_number: Optional[str]

    # Address
    city: Optional[str]
    state: Optional[str]
    country: str

    # Policy
    policy_number: Optional[str]
    policy_type: Optional[str]
    coverage_amount: Optional[float]

    # Risk profile
    risk_category: str
    total_claims_count: int
    total_claims_amount: float

    # Account
    account_status: str
    is_verified: bool
    account_created_at: datetime

    class Config:
        from_attributes = True


class UserProfile(BaseModel):
    """Detailed user profile schema"""
    user_id: UUID
    email: EmailStr
    first_name: str
    last_name: str
    phone_number: Optional[str]
    date_of_birth: Optional[date]

    # Full address
    address: Optional[str]
    city: Optional[str]
    state: Optional[str]
    country: str
    postal_code: Optional[str]

    # Full policy details
    policy_number: Optional[str]
    policy_type: Optional[str]
    policy_start_date: Optional[date]
    policy_end_date: Optional[date]
    premium_amount: Optional[float]
    coverage_amount: Optional[float]

    # Risk profile
    risk_category: str
    total_claims_count: int
    total_claims_amount: float
    fraud_flags_count: int

    # Account metadata
    account_status: str
    is_verified: bool
    is_admin: bool
    account_created_at: datetime
    last_login_at: Optional[datetime]

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """Schema for updating user profile"""
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    phone_number: Optional[str] = Field(None, max_length=20)
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None

    class Config:
        from_attributes = True
