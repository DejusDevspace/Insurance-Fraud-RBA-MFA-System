from sqlalchemy import Column, String, DateTime, Integer, Boolean, Numeric, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, INET
from sqlalchemy.orm import relationship
import datetime as dt
from datetime import datetime
import uuid

from app.database import Base


class TransactionContext(Base):
    __tablename__ = "transaction_contexts"

    # Primary Key
    context_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Keys
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.claim_id", ondelete="CASCADE"), nullable=False,
                      unique=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    device_id = Column(UUID(as_uuid=True), ForeignKey("devices.device_id", ondelete="SET NULL"))

    # Contextual Data
    ip_address = Column(INET)
    geolocation = Column(JSON)  # {country, city, lat, lon}
    is_geolocation_anomaly = Column(Boolean, default=False)
    geolocation_distance_km = Column(Numeric(10, 2))

    # Session Behavior
    session_duration = Column(Integer)  # seconds
    pages_visited = Column(Integer)
    form_fill_time = Column(Integer)  # seconds to fill claim form

    # Timing
    transaction_hour = Column(Integer)  # 0-23
    transaction_day_of_week = Column(Integer)  # 0-6
    is_unusual_time = Column(Boolean, default=False)

    # Pattern Analysis
    days_since_last_claim = Column(Integer)
    is_rapid_succession = Column(Boolean, default=False)
    claims_last_30_days = Column(Integer, default=0)
    claims_last_90_days = Column(Integer, default=0)

    # Device info
    device_type = Column(String(50))
    device_fingerprint = Column(String(255))
    is_trusted_device = Column(Boolean, default=False)
    device_trust_score = Column(Numeric(3, 2))

    # Timestamp
    captured_at = Column(DateTime, default=lambda: datetime.now(dt.UTC), nullable=False)

    # Relationships
    claim = relationship("Claim", back_populates="transaction_context")
    user = relationship("User")
    device = relationship("Device")

    def __repr__(self):
        return f"<TransactionContext(claim_id={self.claim_id}, ip='{self.ip_address}', anomaly={self.is_geolocation_anomaly})>"
