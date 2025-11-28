import api from "./api";

export interface MFAResponse {
    success: boolean;
    message: string;
    claim_status?: string;
}

export interface OTPResponse {
    success: boolean;
    message: string;
    claim_id: string;
    expires_in_minutes: number;
    otp_demo?: string; // Only in demo mode
}

export const mfaService = {
    // Send OTP
    sendOTP: async (claimId: string): Promise<OTPResponse> => {
        const response = await api.post<OTPResponse>("/mfa/send-otp", {
            claim_id: claimId,
        });
        return response.data;
    },

    // Verify OTP
    verifyOTP: async (
        claimId: string,
        otpCode: string
    ): Promise<MFAResponse> => {
        const response = await api.post<MFAResponse>("/mfa/verify-otp", {
            claim_id: claimId,
            otp_code: otpCode,
        });
        return response.data;
    },

    // Verify biometric
    verifyBiometric: async (claimId: string): Promise<MFAResponse> => {
        const response = await api.post<MFAResponse>("/mfa/verify-biometric", {
            claim_id: claimId,
            biometric_verified: true,
        });
        return response.data;
    },

    // Get MFA status
    getMFAStatus: async (claimId: string): Promise<any> => {
        const response = await api.get(`/mfa/status/${claimId}`);
        return response.data;
    },
};
