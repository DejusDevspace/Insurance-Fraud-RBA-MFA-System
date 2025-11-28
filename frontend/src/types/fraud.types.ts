export interface FraudDetection {
    detection_id: string;
    claim_id: string;
    is_suspicious: boolean;
    fraud_probability: number;
    predicted_fraud_type?: string;
    anomaly_score?: number;
    model_used: string;
    detected_at: string;
}

export interface ShapFeature {
    feature: string;
    shap_value: number;
    contribution: "increases" | "decreases";
    magnitude: number;
}

export interface FraudExplanation {
    detection_id: string;
    claim_id: string;
    is_suspicious: boolean;
    fraud_probability: number;
    predicted_fraud_type?: string;
    shap_values: Record<string, number>;
    top_features: ShapFeature[];
    base_value: number;
    explanation: string;
    confidence_level: string;
}
