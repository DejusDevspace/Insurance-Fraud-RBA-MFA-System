import api from "./api";
import type { FraudDetection, FraudExplanation } from "../types/fraud.types";

export const fraudService = {
    // Get fraud detection for claim
    getFraudDetection: async (claimId: string): Promise<FraudDetection> => {
        const response = await api.get<FraudDetection>(`/fraud/${claimId}`);
        return response.data;
    },

    // Get fraud explanation with SHAP
    getFraudExplanation: async (claimId: string): Promise<FraudExplanation> => {
        const response = await api.get<FraudExplanation>(
            `/fraud/${claimId}/explanation`
        );
        return response.data;
    },
};
