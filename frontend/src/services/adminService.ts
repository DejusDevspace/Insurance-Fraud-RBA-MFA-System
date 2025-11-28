import api from "./api";
import type {
    DashboardStats,
    RiskDistribution,
    FraudAlert,
    RecentActivity,
} from "../types/admin.types";

export const adminService = {
    // Get dashboard stats
    getDashboardStats: async (): Promise<DashboardStats> => {
        const response = await api.get<DashboardStats>("/admin/dashboard");
        return response.data;
    },

    // Get risk distribution
    getRiskDistribution: async (): Promise<RiskDistribution> => {
        const response = await api.get<RiskDistribution>(
            "/admin/risk-distribution"
        );
        return response.data;
    },

    // Get fraud alerts
    getFraudAlerts: async (limit: number = 20): Promise<FraudAlert[]> => {
        const response = await api.get<FraudAlert[]>("/admin/fraud-alerts", {
            params: { limit },
        });
        return response.data;
    },

    // Get recent activity
    getRecentActivity: async (
        limit: number = 20
    ): Promise<RecentActivity[]> => {
        const response = await api.get<RecentActivity[]>(
            "/admin/activity/recent",
            {
                params: { limit },
            }
        );
        return response.data;
    },

    // Get all claims with filters
    getAllClaims: async (filters?: {
        status?: string;
        risk_level?: string;
        is_suspicious?: boolean;
        limit?: number;
        offset?: number;
    }): Promise<any> => {
        const response = await api.get("/admin/claims", { params: filters });
        return response.data;
    },
};
