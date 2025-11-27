import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
from sqlalchemy.orm import Session
import logging

from app.models.user import User
from app.models.claim import Claim
from app.models.device import Device

logger = logging.getLogger(__name__)


class FeatureEngineeringService:
    """
    Feature engineering service
    Transforms raw transaction data into features for ML models
    """

    @staticmethod
    def engineer_risk_features(
        claim_data: Dict[str, Any],
        user: User,
        context_data: Dict[str, Any],
        db: Session
    ) -> Dict[str, float]:
        """
        Engineer features for risk scoring model

        Args:
            claim_data: Claim information
            user: User object
            context_data: Transaction context (device, location, timing)
            db: Database session

        Returns:
            Dictionary of engineered features
        """
        features = {}

        # -- Claim amount features
        claim_amount = float(claim_data['claim_amount'])
        coverage_amount = float(user.coverage_amount) if user.coverage_amount else 100000

        features['claim_amount_log'] = np.log1p(claim_amount)
        features['claim_amount_ratio'] = claim_amount / coverage_amount
        features['is_high_claim'] = 1 if (claim_amount / coverage_amount) > 0.5 else 0

        # -- Claim frequency features
        user_claims = db.query(Claim).filter(Claim.user_id == user.user_id).all()

        # Claims in last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        claims_last_30 = len([c for c in user_claims if c.submitted_at >= thirty_days_ago])

        # Claims in last 90 days
        ninety_days_ago = datetime.utcnow() - timedelta(days=90)
        claims_last_90 = len([c for c in user_claims if c.submitted_at >= ninety_days_ago])

        features['claims_last_30_days'] = claims_last_30
        features['claims_last_90_days'] = claims_last_90
        features['has_recent_claims'] = 1 if claims_last_30 > 0 else 0
        features['has_multiple_recent_claims'] = 1 if claims_last_30 > 1 else 0
        features['claims_intensity'] = claims_last_30 / 30.0
        features['user_total_claims'] = len(user_claims)

        # -- Rapid succession
        if user_claims:
            sorted_claims = sorted(user_claims, key=lambda c: c.submitted_at, reverse=True)
            if len(sorted_claims) > 0:
                last_claim = sorted_claims[0]
                days_since_last = (datetime.utcnow() - last_claim.submitted_at).days
            else:
                days_since_last = 999
        else:
            days_since_last = 999

        features['days_since_last_claim_filled'] = days_since_last
        features['is_rapid_succession_numeric'] = 1 if days_since_last < 7 else 0

        # -- Geolocation features
        geolocation_distance = context_data.get('geolocation_distance_km', 0)
        is_geolocation_anomaly = context_data.get('is_geolocation_anomaly', False)

        if geolocation_distance > 500:
            geolocation_risk = 1.0
        elif geolocation_distance > 100:
            geolocation_risk = 0.5
        else:
            geolocation_risk = 0.0

        features['geolocation_distance_km'] = geolocation_distance
        features['geolocation_risk_score'] = geolocation_risk
        features['is_geolocation_anomaly_numeric'] = 1 if is_geolocation_anomaly else 0

        # -- Device features
        device_trust_score = context_data.get('device_trust_score', 0.5)
        is_trusted_device = context_data.get('is_trusted_device', False)

        features['is_untrusted_device'] = 0 if is_trusted_device else 1
        features['device_trust_score'] = device_trust_score
        features['device_risk_score'] = 1.0 - device_trust_score

        # -- Time features
        transaction_hour = context_data.get('transaction_hour', datetime.utcnow().hour)
        transaction_day = context_data.get('transaction_day_of_week', datetime.utcnow().weekday())
        is_unusual_time = context_data.get('is_unusual_time', False)

        features['transaction_hour'] = transaction_hour
        features['transaction_day_of_week'] = transaction_day
        features['is_unusual_time_numeric'] = 1 if is_unusual_time else 0
        features['is_business_hours'] = 1 if 9 <= transaction_hour <= 17 else 0
        features['is_late_night'] = 1 if (transaction_hour >= 22 or transaction_hour <= 5) else 0
        features['is_weekend'] = 1 if transaction_day in [5, 6] else 0

        # -- Session behavior
        form_fill_time = context_data.get('form_fill_time', 300)
        session_duration = context_data.get('session_duration', 600)
        pages_visited = context_data.get('pages_visited', 5)

        features['session_duration_seconds'] = session_duration
        features['form_fill_time_minutes'] = form_fill_time / 60.0
        features['is_rushed_form'] = 1 if form_fill_time < 180 else 0
        features['is_suspicious_session'] = 1 if (form_fill_time < 120 or form_fill_time > 1200) else 0
        features['pages_visited'] = pages_visited

        # -- User features
        account_age = user.account_age_days
        features['account_age_days'] = account_age
        features['account_age_years'] = account_age / 365.0
        features['is_new_account'] = 1 if account_age < 180 else 0

        # User average claim amount
        if user_claims:
            user_avg_claim = sum(float(c.claim_amount) for c in user_claims) / len(user_claims)
        else:
            user_avg_claim = claim_amount
        features['user_avg_claim_amount'] = user_avg_claim

        # -- Document features
        docs_count = claim_data.get('supporting_documents_count', 0)
        features['supporting_documents_count'] = docs_count
        features['has_few_documents'] = 1 if docs_count < 2 else 0

        # --. Policy features
        features['premium_amount'] = float(user.premium_amount) if user.premium_amount else 1000
        features['coverage_amount'] = coverage_amount
        features['premium_ratio'] = features['premium_amount'] / coverage_amount

        # Encode categorical features (using label encoder mappings)
        policy_type_map = {'auto': 0, 'home': 1, 'health': 2, 'life': 3}
        claim_type_map = {'accident': 0, 'theft': 1, 'medical': 2, 'property_damage': 3, 'other': 4}
        risk_category_map = {'low': 0, 'medium': 1, 'high': 2}

        features['policy_type_encoded'] = policy_type_map.get(user.policy_type, 0)
        features['claim_type_encoded'] = claim_type_map.get(claim_data['claim_type'], 4)
        features['risk_category_encoded'] = risk_category_map.get(user.risk_category, 0)

        logger.info(f"Engineered {len(features)} risk features for claim")

        return features

    @staticmethod
    def engineer_fraud_features(
        claim_data: Dict[str, Any],
        user: User,
        context_data: Dict[str, Any],
        db: Session
    ) -> Dict[str, float]:
        """
        Engineer features for fraud detection model

        Args:
            claim_data: Claim information
            user: User object
            context_data: Transaction context
            db: Database session

        Returns:
            Dictionary of engineered features
        """
        features = {}

        # -- Amount-based features (exaggeration indicators)
        claim_amount = float(claim_data['claim_amount'])
        coverage_amount = float(user.coverage_amount) if user.coverage_amount else 100000

        features['claim_amount_log'] = np.log1p(claim_amount)
        features['claim_amount_ratio'] = claim_amount / coverage_amount
        features['is_extreme_amount'] = 1 if (claim_amount / coverage_amount) > 0.7 else 0

        # User average comparison
        user_claims = db.query(Claim).filter(Claim.user_id == user.user_id).all()
        if user_claims:
            user_avg = sum(float(c.claim_amount) for c in user_claims) / len(user_claims)
        else:
            user_avg = claim_amount

        features['amount_vs_user_avg'] = claim_amount / (user_avg + 1)
        features['is_amount_spike'] = 1 if (claim_amount / (user_avg + 1)) > 2.0 else 0
        features['user_avg_claim_amount'] = user_avg

        # Round number (staged indicator)
        features['is_round_amount'] = 1 if (claim_amount % 1000 == 0) else 0

        # -- Duplicate/rapid succession
        if user_claims:
            sorted_claims = sorted(user_claims, key=lambda c: c.submitted_at, reverse=True)
            if len(sorted_claims) > 0:
                days_since_last = (datetime.utcnow() - sorted_claims[0].submitted_at).days
            else:
                days_since_last = 999
        else:
            days_since_last = 999

        features['days_since_last_claim_filled'] = days_since_last
        features['is_rapid_succession_numeric'] = 1 if days_since_last < 7 else 0
        features['is_very_rapid'] = 1 if days_since_last < 3 else 0

        # Claim frequency
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        ninety_days_ago = datetime.utcnow() - timedelta(days=90)

        claims_last_30 = len([c for c in user_claims if c.submitted_at >= thirty_days_ago])
        claims_last_90 = len([c for c in user_claims if c.submitted_at >= ninety_days_ago])

        features['claims_per_month'] = claims_last_30
        features['claims_per_quarter'] = claims_last_90
        features['is_frequent_claimer'] = 1 if claims_last_30 > 2 else 0
        features['user_total_claims'] = len(user_claims)

        # -- Identity theft indicators
        geolocation_distance = context_data.get('geolocation_distance_km', 0)
        is_geolocation_anomaly = context_data.get('is_geolocation_anomaly', False)

        if geolocation_distance > 1000:
            geolocation_risk = 1.0
        elif geolocation_distance > 500:
            geolocation_risk = 0.7
        elif geolocation_distance > 100:
            geolocation_risk = 0.3
        else:
            geolocation_risk = 0.0

        features['geolocation_distance_km'] = geolocation_distance
        features['geolocation_risk'] = geolocation_risk
        features['is_geolocation_anomaly_numeric'] = 1 if is_geolocation_anomaly else 0

        # Device
        device_trust_score = context_data.get('device_trust_score', 0.5)
        is_trusted_device = context_data.get('is_trusted_device', False)

        features['device_trust_score'] = device_trust_score
        features['device_risk'] = 1.0 - device_trust_score
        features['is_untrusted_device'] = 0 if is_trusted_device else 1

        # Identity theft signal
        features['identity_theft_signal'] = (
            features['is_geolocation_anomaly_numeric'] * 0.6 +
            features['is_untrusted_device'] * 0.4
        )

        # -- Staged incident indicators
        docs_count = claim_data.get('supporting_documents_count', 0)
        form_fill_time = context_data.get('form_fill_time', 300)
        is_unusual_time = context_data.get('is_unusual_time', False)

        features['supporting_documents_count'] = docs_count
        features['has_few_documents'] = 1 if docs_count < 2 else 0
        features['form_fill_time_seconds'] = form_fill_time
        features['form_fill_time_minutes'] = form_fill_time / 60.0
        features['is_rushed_form'] = 1 if form_fill_time < 180 else 0

        # Staged signal
        features['staged_signal'] = (
            features['is_round_amount'] * 0.3 +
            features['has_few_documents'] * 0.4 +
            features['is_rushed_form'] * 0.3
        )

        # -- Out-of-pattern indicators
        account_age = user.account_age_days
        features['account_age_days'] = account_age
        features['is_new_account'] = 1 if account_age < 180 else 0
        features['claim_frequency_anomaly'] = 1 if (len(user_claims) > account_age / 60) else 0

        # Session behavior
        session_duration = context_data.get('session_duration', 600)
        features['session_duration_seconds'] = session_duration
        features['is_suspicious_session'] = 1 if (form_fill_time < 120 or form_fill_time > 1200) else 0

        # -- Time features
        transaction_hour = context_data.get('transaction_hour', datetime.utcnow().hour)
        transaction_day = context_data.get('transaction_day_of_week', datetime.utcnow().weekday())

        features['transaction_hour'] = transaction_hour
        features['transaction_day_of_week'] = transaction_day
        features['is_unusual_time_numeric'] = 1 if is_unusual_time else 0
        features['is_late_night'] = 1 if (transaction_hour >= 22 or transaction_hour <= 5) else 0
        features['is_weekend'] = 1 if transaction_day in [5, 6] else 0

        # -- Policy features
        features['premium_amount'] = float(user.premium_amount) if user.premium_amount else 1000
        features['coverage_amount'] = coverage_amount

        # Categorical encoding
        policy_type_map = {'auto': 0, 'home': 1, 'health': 2, 'life': 3}
        claim_type_map = {'accident': 0, 'theft': 1, 'medical': 2, 'property_damage': 3, 'other': 4}
        risk_category_map = {'low': 0, 'medium': 1, 'high': 2}
        claim_status_map = {'pending': 0, 'approved': 1, 'rejected': 2, 'under_review': 3}

        features['policy_type_encoded'] = policy_type_map.get(user.policy_type, 0)
        features['claim_type_encoded'] = claim_type_map.get(claim_data['claim_type'], 4)
        features['risk_category_encoded'] = risk_category_map.get(user.risk_category, 0)
        features['claim_status_encoded'] = claim_status_map.get('pending', 0)

        logger.info(f"Engineered {len(features)} fraud features for claim")

        return features
