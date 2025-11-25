from sqlalchemy import Column, String, DateTime, Boolean, Integer, Numeric, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import datetime as dt
from datetime import datetime
import uuid

from app.database import Base


class Device(Base):
    __tablename__ = "devices"

    # Primary Key
    device_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Key
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)

    # Device Fingerprint
    device_fingerprint = Column(String(255), unique=True, nullable=False, index=True)
    device_type = Column(String(50))  # mobile, desktop, tablet
    os = Column(String(100))
    browser = Column(String(100))
    browser_version = Column(String(50))

    # Trust Level
    is_trusted = Column(Boolean, default=False, nullable=False)
    device_trust_score = Column(Numeric(3, 2), default=0.5)  # 0.00 to 1.00
    first_seen_at = Column(DateTime, default=lambda: datetime.now(dt.UTC), nullable=False)
    last_seen_at = Column(DateTime, default=lambda: datetime.now(dt.UTC), nullable=False)
    usage_count = Column(Integer, default=1, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(dt.UTC), nullable=False)

    # Relationships
    user = relationship("User", back_populates="devices")

    def __repr__(self):
        return f"<Device(fingerprint='{self.device_fingerprint[:16]}...', type='{self.device_type}', trusted={self.is_trusted})>"

    def update_usage(self):
        """Update device usage statistics"""
        self.last_seen_at = datetime.now(dt.UTC)
        self.usage_count += 1

        # Increase trust score based on usage
        if self.usage_count > 5 and not self.is_trusted:
            self.is_trusted = True
            self.device_trust_score = min(0.95, self.device_trust_score + 0.2)
