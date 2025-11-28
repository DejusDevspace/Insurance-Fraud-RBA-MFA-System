export const CLAIM_TYPES = [
    { value: "accident", label: "Accident" },
    { value: "theft", label: "Theft" },
    { value: "medical", label: "Medical" },
    { value: "property_damage", label: "Property Damage" },
    { value: "other", label: "Other" },
] as const;

export const CLAIM_STATUS = {
    PENDING: "pending",
    APPROVED: "approved",
    REJECTED: "rejected",
    UNDER_REVIEW: "under_review",
} as const;

export const RISK_LEVELS = {
    LOW: "low",
    MEDIUM: "medium",
    HIGH: "high",
} as const;

export const MFA_METHODS = {
    OTP: "otp",
    BIOMETRIC: "biometric",
} as const;

export const POLICY_TYPES = [
    { value: "auto", label: "Auto Insurance" },
    { value: "home", label: "Home Insurance" },
    { value: "health", label: "Health Insurance" },
    { value: "life", label: "Life Insurance" },
] as const;

export const ROUTES = {
    HOME: "/",
    LOGIN: "/login",
    REGISTER: "/register",
    DASHBOARD: "/dashboard",
    SUBMIT_CLAIM: "/submit-claim",
    CLAIMS_HISTORY: "/claims",
    CLAIM_DETAILS: "/claims/:claimId",
    ADMIN_DASHBOARD: "/admin",
} as const;

export const API_ROUTES = {
    AUTH: {
        REGISTER: "/auth/register",
        LOGIN: "/auth/login",
        ME: "/auth/me",
        LOGOUT: "/auth/logout",
    },
    CLAIMS: {
        LIST: "/claims",
        SUBMIT: "/claims",
        DETAIL: (id: string) => `/claims/${id}`,
        STATUS: (id: string) => `/claims/${id}/status`,
    },
    RISK: {
        SCORE: (id: string) => `/risk/${id}`,
        EXPLANATION: (id: string) => `/risk/${id}/explanation`,
        HISTORY: "/risk/user/history",
    },
    FRAUD: {
        DETECTION: (id: string) => `/fraud/${id}`,
        EXPLANATION: (id: string) => `/fraud/${id}/explanation`,
    },
    MFA: {
        SEND_OTP: "/mfa/send-otp",
        VERIFY_OTP: "/mfa/verify-otp",
        VERIFY_BIOMETRIC: "/mfa/verify-biometric",
        STATUS: (id: string) => `/mfa/status/${id}`,
    },
    ADMIN: {
        DASHBOARD: "/admin/dashboard",
        RISK_DISTRIBUTION: "/admin/risk-distribution",
        FRAUD_ALERTS: "/admin/fraud-alerts",
        RECENT_ACTIVITY: "/admin/activity/recent",
        ALL_CLAIMS: "/admin/claims",
    },
} as const;
