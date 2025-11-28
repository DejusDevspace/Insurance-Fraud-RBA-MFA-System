export interface User {
    user_id: string;
    email: string;
    first_name: string;
    last_name: string;
    phone_number?: string;
    city?: string;
    state?: string;
    country: string;
    policy_number?: string;
    policy_type?: string;
    coverage_amount?: number;
    risk_category: string;
    total_claims_count: number;
    total_claims_amount: number;
    account_status: string;
    is_verified: boolean;
    account_created_at: string;
    is_admin?: boolean;
}

export interface UserProfile extends User {
    date_of_birth?: string;
    address?: string;
    postal_code?: string;
    policy_start_date?: string;
    policy_end_date?: string;
    premium_amount?: number;
    fraud_flags_count: number;
    last_login_at?: string;
}
