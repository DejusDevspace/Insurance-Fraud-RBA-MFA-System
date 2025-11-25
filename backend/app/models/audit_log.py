from sqlalchemy import Column, String, DateTime, ForeignKey, JSON, Text
from sqlalchemy.dialects.postgresql import UUID, INET
from sqlalchemy.orm import relationship
import datetime as dt
from datetime import datetime
import uuid

from app.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    # Primary Key
    log_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Keys (nullable for system actions)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="SET NULL"))
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.claim_id", ondelete="SET NULL"))

    # Action Details
    action_type = Column(String(100), nullable=False, index=True)
    # login, claim_submit, risk_calculated, fraud_detected, mfa_triggered, etc.
    action_result = Column(String(50))  # success, failure, pending

    # Actor
    actor_type = Column(String(50), default="user")  # user, system, admin
    actor_id = Column(UUID(as_uuid=True))

    # Additional Context
    details = Column(JSON)  # Flexible field for action-specific data
    ip_address = Column(INET)
    user_agent = Column(Text)

    # Timestamp
    timestamp = Column(DateTime, default=lambda: datetime.now(dt.UTC), nullable=False, index=True)

    # Relationships
    user = relationship("User")
    claim = relationship("Claim")

    def __repr__(self):
        return f"<AuditLog(action='{self.action_type}', result='{self.action_result}', timestamp={self.timestamp})>"
