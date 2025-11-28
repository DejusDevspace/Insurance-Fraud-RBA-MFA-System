export type RiskLevel = "low" | "medium" | "high";

export interface RiskScore {
    risk_score_id: string;
    claim_id: string;
    risk_score: number;
    risk_level: RiskLevel;
    factors: Record<string, number>;
    model_version?: string;
    calculation_method: string;
    calculated_at: string;
}

export interface RiskFactor {
    factor: string;
    score: number;
}

export interface RiskAssessment {
    risk_score: number;
    risk_level: RiskLevel;
    factors: Record<string, number>;
    top_risk_factors: RiskFactor[];
    requires_mfa: boolean;
    mfa_method?: string;
    explanation: string;
}
