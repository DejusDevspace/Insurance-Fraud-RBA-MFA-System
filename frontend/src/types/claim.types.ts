export type ClaimType =
    | "accident"
    | "theft"
    | "medical"
    | "property_damage"
    | "other";
export type ClaimStatus = "pending" | "approved" | "rejected" | "under_review";

export interface ClaimSubmission {
    claim_type: ClaimType;
    claim_amount: number;
    incident_date: string;
    claim_description: string;
    supporting_documents_count?: number;
    session_duration: number;
    pages_visited: number;
    form_fill_time: number;
    
    // Optional Demo Overrides
    device_trust_score?: number;
    is_trusted_device?: boolean;
    geolocation_distance_km?: number;
    is_geolocation_anomaly?: boolean;
}

export interface Claim {
    claim_id: string;
    claim_number: string;
    claim_type: ClaimType;
    claim_amount: number;
    incident_date: string;
    claim_description?: string;
    claim_status: ClaimStatus;
    submitted_at: string;
    processed_at?: string;
    requires_mfa?: boolean;
    mfa_method?: string;
    risk_level?: string;
    fraud_probability?: number;
    is_suspicious?: boolean;
    approval_status?: string;
    rejection_reason?: string;
    approved_amount?: number;
    supporting_documents_count?: number;
}

export interface ClaimSubmissionResponse {
    claim: Claim;
    risk_assessment: {
        risk_score: number;
        risk_level: string;
        fraud_probability: number;
    };
    requires_mfa: boolean;
    mfa_method?: string;
    message: string;
}
