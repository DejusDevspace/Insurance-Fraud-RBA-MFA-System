from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, INET
from sqlalchemy.orm import relationship
import datetime as dt
from datetime import datetime
import uuid

from app.database import Base


class AuthenticationEvent(Base):
    __tablename__ = "authentication_events"

    # Primary Key
    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Keys
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.claim_id", ondelete="SET NULL"))

    # Event Details
    event_type = Column(String(50), nullable=False, index=True)
    # login, claim_submission, mfa_trigger, mfa_success, mfa_failure
    auth_method = Column(String(50))  # password, otp, biometric
    auth_result = Column(String(20), nullable=False)  # success, failure, pending

    # Context
    risk_level_at_auth = Column(String(20))  # low, medium, high
    mfa_required = Column(Boolean, default=False)
    mfa_completed = Column(Boolean, default=False)

    # Session Info
    session_id = Column(String(255))
    ip_address = Column(INET)
    user_agent = Column(Text)

    # Timestamp
    event_timestamp = Column(DateTime, default=lambda: datetime.now(dt.UTC), nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="authentication_events")
    claim = relationship("Claim", back_populates="authentication_events")

    def __repr__(self):
        return f"<AuthenticationEvent(type='{self.event_type}', result='{self.auth_result}', timestamp={self.event_timestamp})>"
