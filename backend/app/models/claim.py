from sqlalchemy import Column, String, Date, DateTime, Numeric, Integer, Boolean, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import datetime as dt
from datetime import datetime
import uuid

from app.database import Base


class Claim(Base):
    __tablename__ = "claims"

    # Primary Key
    claim_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Key
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)

    # Claim Details
    claim_number = Column(String(50), unique=True, nullable=False, index=True)
    claim_type = Column(String(50), nullable=False)  # accident, theft, medical, property_damage, other
    claim_amount = Column(Numeric(12, 2), nullable=False)
    incident_date = Column(Date, nullable=False)
    claim_description = Column(Text, nullable=False)
    supporting_documents_count = Column(Integer, default=0)

    # Status Tracking
    claim_status = Column(String(20), default="pending", nullable=False)  # pending, approved, rejected, under_review
    submitted_at = Column(DateTime, default=lambda: datetime.now(dt.UTC), nullable=False, index=True)
    processed_at = Column(DateTime)

    # Fraud Indicators (Ground Truth - for evaluation/training)
    is_fraudulent = Column(Boolean, default=False, nullable=False, index=True)
    fraud_type = Column(String(50))  # exaggerated, duplicate, staged, identity_theft, out_of_pattern
    fraud_confidence_score = Column(Numeric(5, 4))  # 0.0000 to 1.0000

    # Decision
    approval_status = Column(String(20))  # approved, rejected, flagged
    rejection_reason = Column(Text)
    approved_amount = Column(Numeric(12, 2))

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(dt.UTC), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(dt.UTC), onupdate=lambda: datetime.now(dt.UTC), nullable=False)

    # Relationships
    user = relationship("User", back_populates="claims")
    risk_scores = relationship("RiskScore", back_populates="claim", cascade="all, delete-orphan")
    fraud_detections = relationship("FraudDetection", back_populates="claim", cascade="all, delete-orphan")
    transaction_context = relationship("TransactionContext", back_populates="claim", uselist=False,
                                       cascade="all, delete-orphan")
    authentication_events = relationship("AuthenticationEvent", back_populates="claim")

    def __repr__(self):
        return f"<Claim(claim_number='{self.claim_number}', amount={self.claim_amount}, status='{self.claim_status}')>"

    @property
    def days_since_submission(self):
        """Calculate days since claim submission"""
        return (datetime.now(dt.UTC) - self.submitted_at).days

    @property
    def is_processed(self):
        """Check if claim has been processed"""
        return self.processed_at is not None
