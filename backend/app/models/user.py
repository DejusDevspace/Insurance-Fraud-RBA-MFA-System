from sqlalchemy import Column, String, Date, DateTime, Numeric, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import datetime as dt
from datetime import datetime
import uuid

from app.database import Base


class User(Base):
    __tablename__ = "users"

    # Primary Key
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Authentication
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)

    # Personal Information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    phone_number = Column(String(20))
    date_of_birth = Column(Date)

    # Address
    address = Column(String)
    city = Column(String(100))
    state = Column(String(100))
    country = Column(String(100), default="USA")
    postal_code = Column(String(20))

    # Account Metadata
    account_created_at = Column(DateTime, default=lambda: datetime.now(dt.UTC), nullable=False)
    last_login_at = Column(DateTime)
    account_status = Column(String(20), default="active", nullable=False)  # active, suspended, closed
    is_verified = Column(Boolean, default=False, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)

    # Insurance Policy Information
    policy_number = Column(String(50), unique=True, index=True)
    policy_type = Column(String(50))  # auto, health, home, life
    policy_start_date = Column(Date)
    policy_end_date = Column(Date)
    premium_amount = Column(Numeric(10, 2))
    coverage_amount = Column(Numeric(12, 2))

    # Risk Profile
    risk_category = Column(String(20), default="low", nullable=False)  # low, medium, high
    total_claims_count = Column(Integer, default=0, nullable=False)
    total_claims_amount = Column(Numeric(12, 2), default=0, nullable=False)
    fraud_flags_count = Column(Integer, default=0, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(dt.UTC), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(dt.UTC), onupdate=lambda: datetime.now(dt.UTC), nullable=False)

    # Relationships
    claims = relationship("Claim", back_populates="user", cascade="all, delete-orphan")
    devices = relationship("Device", back_populates="user", cascade="all, delete-orphan")
    authentication_events = relationship("AuthenticationEvent", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(email='{self.email}', policy_number='{self.policy_number}')>"

    @property
    def full_name(self):
        """Get user's full name"""
        return f"{self.first_name} {self.last_name}"

    @property
    def account_age_days(self):
        """Calculate account age in days"""
        return (datetime.now(dt.UTC)- self.account_created_at).days
