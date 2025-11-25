from sqlalchemy import Column, String, DateTime, Numeric, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import datetime as dt
from datetime import datetime
import uuid

from app.database import Base


class RiskScore(Base):
    __tablename__ = "risk_scores"

    # Primary Key
    risk_score_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Keys
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.claim_id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)

    # Risk Calculation
    risk_score = Column(Numeric(5, 4), nullable=False)  # 0.0000 to 1.0000
    risk_level = Column(String(20), nullable=False, index=True)  # low, medium, high

    # Contributing Factors (for explainability)
    factors = Column(JSON)  # Store individual factor scores as JSON
    # Example: {"high_claim_amount": 0.8, "multiple_recent_claims": 0.6, ...}

    # Model Information
    model_version = Column(String(20))
    calculation_method = Column(String(50))  # rule_based, ml_enhanced

    # Timestamp
    calculated_at = Column(DateTime, default=lambda: datetime.now(dt.UTC), nullable=False)

    # Relationships
    claim = relationship("Claim", back_populates="risk_scores")
    user = relationship("User")

    def __repr__(self):
        return f"<RiskScore(claim_id={self.claim_id}, risk_level='{self.risk_level}', score={self.risk_score})>"
