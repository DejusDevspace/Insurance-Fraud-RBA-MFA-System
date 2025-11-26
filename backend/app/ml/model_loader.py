import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton class to load and cache ML models
    Models are loaded once at startup and reused across requests
    """

    _instance: Optional['ModelLoader'] = None
    _models_loaded: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._models_loaded:
            self.models_path = Path(settings.MODELS_PATH)
            self.models: Dict[str, Any] = {}
            self.feature_names: Dict[str, list] = {}
            self.label_encoders: Dict[str, Any] = {}
            self.scalers: Dict[str, Any] = {}
            self.explainers: Dict[str, Any] = {}
            self.metadata: Dict[str, Any] = {}

            self._load_all_models()
            ModelLoader._models_loaded = True

    def _load_all_models(self):
        """Load all required models and artifacts"""
        logger.info("Loading ML models...")

        try:
            # Load Risk Model
            self._load_risk_model()

            # Load Fraud Detection Models
            self._load_fraud_models()

            # Load SHAP Explainers
            self._load_explainers()

            logger.info("✓ All ML models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load ML models: {str(e)}")
            raise

    def _load_risk_model(self):
        """Load risk scoring model and artifacts"""
        try:
            # Load model
            risk_model_path = self.models_path / "risk_model.pkl"
            self.models['risk_model'] = joblib.load(risk_model_path)
            logger.info("✓ Loaded risk_model.pkl")

            # Load scaler
            risk_scaler_path = self.models_path / "risk_scaler.pkl"
            self.scalers['risk_scaler'] = joblib.load(risk_scaler_path)
            logger.info("✓ Loaded risk_scaler.pkl")

            # Load feature names
            risk_features_path = self.models_path / "risk_feature_names.json"
            with open(risk_features_path, 'r') as f:
                self.feature_names['risk_features'] = json.load(f)
            logger.info("✓ Loaded risk_feature_names.json")

            # Load label encoders
            risk_encoders_path = self.models_path / "risk_label_encoders.pkl"
            self.label_encoders['risk_encoders'] = joblib.load(risk_encoders_path)
            logger.info("✓ Loaded risk_label_encoders.pkl")

            # Load metadata
            risk_metadata_path = self.models_path / "risk_model_metadata.json"
            with open(risk_metadata_path, 'r') as f:
                self.metadata['risk_model'] = json.load(f)
            logger.info("✓ Loaded risk_model_metadata.json")

        except Exception as e:
            logger.error(f"Failed to load risk model: {str(e)}")
            raise

    def _load_fraud_models(self):
        """Load fraud detection models and artifacts"""
        try:
            # Load Isolation Forest (unsupervised)
            iso_forest_path = self.models_path / "isolation_forest.pkl"
            self.models['isolation_forest'] = joblib.load(iso_forest_path)
            logger.info("✓ Loaded isolation_forest.pkl")

            # Load XGBoost Classifier (supervised)
            fraud_classifier_path = self.models_path / "fraud_classifier.pkl"
            self.models['fraud_classifier'] = joblib.load(fraud_classifier_path)
            logger.info("✓ Loaded fraud_classifier.pkl")

            # Load scaler
            fraud_scaler_path = self.models_path / "fraud_scaler.pkl"
            self.scalers['fraud_scaler'] = joblib.load(fraud_scaler_path)
            logger.info("✓ Loaded fraud_scaler.pkl")

            # Load feature names
            fraud_features_path = self.models_path / "fraud_feature_names.json"
            with open(fraud_features_path, 'r') as f:
                self.feature_names['fraud_features'] = json.load(f)
            logger.info("✓ Loaded fraud_feature_names.json")

            # Load label encoders
            fraud_encoders_path = self.models_path / "fraud_label_encoders.pkl"
            self.label_encoders['fraud_encoders'] = joblib.load(fraud_encoders_path)
            logger.info("✓ Loaded fraud_label_encoders.pkl")

            # Load metadata
            fraud_metadata_path = self.models_path / "fraud_model_metadata.json"
            with open(fraud_metadata_path, 'r') as f:
                self.metadata['fraud_model'] = json.load(f)
            logger.info("✓ Loaded fraud_model_metadata.json")

        except Exception as e:
            logger.error(f"Failed to load fraud models: {str(e)}")
            raise

    def _load_explainers(self):
        """Load SHAP explainers"""
        try:
            # Load Risk SHAP explainer
            risk_explainer_path = self.models_path / "risk_shap_explainer.pkl"
            self.explainers['risk_explainer'] = joblib.load(risk_explainer_path)
            logger.info("✓ Loaded risk_shap_explainer.pkl")

            # Load Fraud SHAP explainer
            fraud_explainer_path = self.models_path / "fraud_shap_explainer.pkl"
            self.explainers['fraud_explainer'] = joblib.load(fraud_explainer_path)
            logger.info("✓ Loaded fraud_shap_explainer.pkl")

        except Exception as e:
            logger.error(f"Failed to load SHAP explainers: {str(e)}")
            raise

    def get_risk_model(self):
        """Get risk scoring model"""
        return self.models.get('risk_model')

    def get_fraud_classifier(self):
        """Get fraud classification model"""
        return self.models.get('fraud_classifier')

    def get_isolation_forest(self):
        """Get isolation forest model"""
        return self.models.get('isolation_forest')

    def get_risk_scaler(self):
        """Get risk model scaler"""
        return self.scalers.get('risk_scaler')

    def get_fraud_scaler(self):
        """Get fraud model scaler"""
        return self.scalers.get('fraud_scaler')

    def get_risk_features(self):
        """Get risk model feature names"""
        return self.feature_names.get('risk_features')

    def get_fraud_features(self):
        """Get fraud model feature names"""
        return self.feature_names.get('fraud_features')

    def get_risk_encoders(self):
        """Get risk model label encoders"""
        return self.label_encoders.get('risk_encoders')

    def get_fraud_encoders(self):
        """Get fraud model label encoders"""
        return self.label_encoders.get('fraud_encoders')

    def get_risk_explainer(self):
        """Get risk SHAP explainer"""
        return self.explainers.get('risk_explainer')

    def get_fraud_explainer(self):
        """Get fraud SHAP explainer"""
        return self.explainers.get('fraud_explainer')

    def get_risk_metadata(self):
        """Get risk model metadata"""
        return self.metadata.get('risk_model', {})

    def get_fraud_metadata(self):
        """Get fraud model metadata"""
        return self.metadata.get('fraud_model', {})


# Global instance
_model_loader_instance: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """
    Get global ModelLoader instance
    This should be called after models are loaded at startup
    """
    global _model_loader_instance
    if _model_loader_instance is None:
        _model_loader_instance = ModelLoader()
    return _model_loader_instance
