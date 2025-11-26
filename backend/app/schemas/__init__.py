from app.schemas.user import (
    UserBase,
    UserCreate,
    UserLogin,
    UserResponse,
    UserProfile,
    UserUpdate
)
from app.schemas.auth import (
    Token,
    TokenData,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse
)
from app.schemas.claim import (
    ClaimBase,
    ClaimCreate,
    ClaimResponse,
    ClaimListResponse,
    ClaimDetailResponse,
    ClaimUpdate
)
from app.schemas.risk import (
    RiskScoreResponse,
    RiskFactorsResponse,
    RiskAssessmentResponse
)
from app.schemas.fraud import (
    FraudDetectionResponse,
    FraudExplanationResponse,
    ShapFeature
)
from app.schemas.admin import (
    DashboardStatsResponse,
    RiskDistributionResponse,
    FraudAlertResponse,
    UserRiskProfileResponse
)

__all__ = [
    # User
    "UserBase",
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "UserProfile",
    "UserUpdate",
    # Auth
    "Token",
    "TokenData",
    "LoginRequest",
    "LoginResponse",
    "RegisterRequest",
    "RegisterResponse",
    # Claim
    "ClaimBase",
    "ClaimCreate",
    "ClaimResponse",
    "ClaimListResponse",
    "ClaimDetailResponse",
    "ClaimUpdate",
    # Risk
    "RiskScoreResponse",
    "RiskFactorsResponse",
    "RiskAssessmentResponse",
    # Fraud
    "FraudDetectionResponse",
    "FraudExplanationResponse",
    "ShapFeature",
    # Admin
    "DashboardStatsResponse",
    "RiskDistributionResponse",
    "FraudAlertResponse",
    "UserRiskProfileResponse",
]
