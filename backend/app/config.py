from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "Insurance Fraud Detection System"
    DEBUG: bool = True
    VERSION: str = "1.0.0"

    # Database
    DATABASE_URL: str

    # JWT Authentication
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # ML Models
    MODELS_PATH: str = "./models"

    # MFA
    OTP_EXPIRY_MINUTES: int = 5
    OTP_LENGTH: int = 6

    # Risk Thresholds
    RISK_LOW_THRESHOLD: float = 0.35
    RISK_HIGH_THRESHOLD: float = 0.65

    # Fraud Thresholds
    FRAUD_PROBABILITY_THRESHOLD: float = 0.5
    FRAUD_HIGH_CONFIDENCE_THRESHOLD: float = 0.7

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    Using lru_cache ensures settings are loaded only once
    """
    return Settings()


# Convenience access
settings = get_settings()
