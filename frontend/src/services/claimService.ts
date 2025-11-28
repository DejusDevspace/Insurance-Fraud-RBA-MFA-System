import api from "./api";
import type {
    ClaimSubmission,
    Claim,
    ClaimSubmissionResponse,
} from "../types/claim.types";

export const claimService = {
    // Submit new claim
    submitClaim: async (
        data: ClaimSubmission
    ): Promise<ClaimSubmissionResponse> => {
        const response = await api.post<ClaimSubmissionResponse>(
            "/claims",
            data
        );
        return response.data;
    },

    // Get user's claims
    getUserClaims: async (
        limit: number = 50,
        offset: number = 0
    ): Promise<Claim[]> => {
        const response = await api.get<Claim[]>("/claims", {
            params: { limit, offset },
        });
        return response.data;
    },

    // Get claim by ID
    getClaimById: async (claimId: string): Promise<Claim> => {
        const response = await api.get<Claim>(`/claims/${claimId}`);
        return response.data;
    },

    // Get claim status
    getClaimStatus: async (claimId: string): Promise<any> => {
        const response = await api.get(`/claims/${claimId}/status`);
        return response.data;
    },
};
