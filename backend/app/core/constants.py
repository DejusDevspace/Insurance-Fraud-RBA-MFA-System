from enum import Enum


class RiskLevel(str, Enum):
    """Risk classification levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FraudType(str, Enum):
    """Types of fraud patterns"""
    EXAGGERATED = "exaggerated"
    DUPLICATE = "duplicate"
    STAGED = "staged"
    IDENTITY_THEFT = "identity_theft"
    OUT_OF_PATTERN = "out_of_pattern"


class ClaimType(str, Enum):
    """Types of insurance claims"""
    ACCIDENT = "accident"
    THEFT = "theft"
    MEDICAL = "medical"
    PROPERTY_DAMAGE = "property_damage"
    OTHER = "other"


class ClaimStatus(str, Enum):
    """Claim processing status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"


class PolicyType(str, Enum):
    """Insurance policy types"""
    AUTO = "auto"
    HOME = "home"
    HEALTH = "health"
    LIFE = "life"


class AccountStatus(str, Enum):
    """User account status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


class AuthEventType(str, Enum):
    """Authentication event types"""
    LOGIN = "login"
    LOGOUT = "logout"
    CLAIM_SUBMISSION = "claim_submission"
    MFA_TRIGGER = "mfa_trigger"
    MFA_SUCCESS = "mfa_success"
    MFA_FAILURE = "mfa_failure"


class MFAMethod(str, Enum):
    """Multi-factor authentication methods"""
    PASSWORD = "password"
    OTP = "otp"
    BIOMETRIC = "biometric"


class DeviceType(str, Enum):
    """Device types"""
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"


# Risk scoring weights
RISK_WEIGHTS = {
    "fraud_probability": 0.40,
    "claim_amount_ratio": 0.15,
    "recent_claims": 0.15,
    "geolocation_anomaly": 0.15,
    "device_trust": 0.10,
    "time_anomaly": 0.05
}

# Authentication thresholds
MFA_REQUIRED_RISK_SCORE = 0.35  # Require MFA if risk > 0.35
BIOMETRIC_REQUIRED_RISK_SCORE = 0.65  # Require biometric if risk > 0.65

# Claim limits
MAX_CLAIM_AMOUNT = 1_000_000  # Maximum claim amount
MIN_CLAIM_AMOUNT = 1  # Minimum claim amount

# Rate limiting
MAX_LOGIN_ATTEMPTS = 5
LOGIN_ATTEMPT_WINDOW_MINUTES = 15

# Session settings
SESSION_TIMEOUT_MINUTES = 30
