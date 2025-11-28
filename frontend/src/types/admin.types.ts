export interface DashboardStats {
    total_claims: number;
    pending_claims: number;
    approved_claims: number;
    rejected_claims: number;
    fraud_detected: number;
    fraud_rate: number;
    high_risk_claims: number;
    medium_risk_claims: number;
    low_risk_claims: number;
    total_claim_amount: number;
    avg_claim_amount: number;
    mfa_triggered: number;
    mfa_success_rate: number;
    active_users: number;
    new_users_today: number;
}

export interface RiskDistribution {
    low_risk_count: number;
    medium_risk_count: number;
    high_risk_count: number;
    low_risk_percentage: number;
    medium_risk_percentage: number;
    high_risk_percentage: number;
    average_risk_score: number;
}

export interface FraudAlert {
    detection_id: string;
    claim_id: string;
    claim_number: string;
    user_id: string;
    user_email: string;
    fraud_probability: number;
    predicted_fraud_type?: string;
    claim_amount: number;
    detected_at: string;
}

export interface RecentActivity {
    claim_id: string;
    claim_number: string;
    user_email: string;
    claim_type: string;
    claim_amount: number;
    risk_level: string;
    fraud_probability?: number;
    submitted_at: string;
    status: string;
}
