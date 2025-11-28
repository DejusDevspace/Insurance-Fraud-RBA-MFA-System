import api from "./api";
import type { RiskScore, RiskAssessment } from "../types/risk.types";

export const riskService = {
    // Get risk score for claim
    getRiskScore: async (claimId: string): Promise<RiskScore> => {
        const response = await api.get<RiskScore>(`/risk/${claimId}`);
        return response.data;
    },

    // Get risk explanation
    getRiskExplanation: async (claimId: string): Promise<RiskAssessment> => {
        const response = await api.get<RiskAssessment>(
            `/risk/${claimId}/explanation`
        );
        return response.data;
    },

    // Get user's risk history
    getUserRiskHistory: async (limit: number = 10): Promise<any> => {
        const response = await api.get("/risk/user/history", {
            params: { limit },
        });
        return response.data;
    },
};
