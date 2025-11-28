import api from "./api";
import type {
    LoginRequest,
    RegisterRequest,
    AuthResponse,
    RegisterResponse,
} from "../types/auth.types";
import type { User } from "../types/user.types";

export const authService = {
    // Register new user
    register: async (data: RegisterRequest): Promise<RegisterResponse> => {
        const response = await api.post<RegisterResponse>(
            "/auth/register",
            data
        );
        return response.data;
    },

    // Login
    login: async (data: LoginRequest): Promise<AuthResponse> => {
        const response = await api.post<AuthResponse>("/auth/login", data);
        return response.data;
    },

    // Get current user
    getCurrentUser: async (): Promise<User> => {
        const response = await api.get<User>("/auth/me");
        return response.data;
    },

    // Logout
    logout: async (): Promise<void> => {
        await api.post("/auth/logout");
        // Clear local storage
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");
        localStorage.removeItem("user");
    },

    // Store auth data in localStorage
    storeAuthData: (data: AuthResponse) => {
        localStorage.setItem("access_token", data.access_token);
        localStorage.setItem("refresh_token", data.refresh_token);
        localStorage.setItem("user", JSON.stringify(data.user));
    },

    // Get stored user
    getStoredUser: (): User | null => {
        const userStr = localStorage.getItem("user");
        if (userStr) {
            try {
                return JSON.parse(userStr);
            } catch {
                return null;
            }
        }
        return null;
    },

    // Check if user is authenticated
    isAuthenticated: (): boolean => {
        return !!localStorage.getItem("access_token");
    },
};
