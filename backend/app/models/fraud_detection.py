from sqlalchemy import Column, String, DateTime, Numeric, Boolean, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import datetime as dt
from datetime import datetime
import uuid

from app.database import Base


class FraudDetection(Base):
    __tablename__ = "fraud_detections"

    # Primary Key
    detection_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Keys
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.claim_id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)

    # Detection Results
    is_suspicious = Column(Boolean, nullable=False, index=True)
    fraud_probability = Column(Numeric(5, 4), nullable=False)  # 0.0000 to 1.0000
    predicted_fraud_type = Column(String(50))

    # Model Outputs
    anomaly_score = Column(Numeric(10, 6))
    model_used = Column(String(50))  # isolation_forest, supervised_classifier, ensemble
    model_version = Column(String(20))

    # Explainability Data
    shap_values = Column(JSON)  # Store SHAP values for features
    top_contributing_features = Column(JSON)  # Top 5 features affecting prediction
    # Example: [{"feature": "claim_amount_ratio", "importance": 0.85, "contribution": "increases"}, ...]

    # Timestamp
    detected_at = Column(DateTime, default=lambda: datetime.now(dt.UTC), nullable=False)

    # Relationships
    claim = relationship("Claim", back_populates="fraud_detections")
    user = relationship("User")

    def __repr__(self):
        return f"<FraudDetection(claim_id={self.claim_id}, suspicious={self.is_suspicious}, probability={self.fraud_probability})>"
