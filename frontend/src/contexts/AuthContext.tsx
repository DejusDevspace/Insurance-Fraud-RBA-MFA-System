import React, { createContext, useState, useEffect } from "react";
import type { ReactNode } from "react";
import type { User } from "../types/user.types";
import type { LoginRequest, RegisterRequest } from "../types/auth.types";
import { authService } from "../services/authService";
import { handleApiError } from "../services/api";

interface AuthContextType {
    user: User | null;
    isAuthenticated: boolean;
    isLoading: boolean;
    login: (credentials: LoginRequest) => Promise<void>;
    register: (data: RegisterRequest) => Promise<void>;
    logout: () => Promise<void>;
    refreshUser: () => Promise<void>;
}

export const AuthContext = createContext<AuthContextType | undefined>(
    undefined
);

interface AuthProviderProps {
    children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
    const [user, setUser] = useState<User | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    // Initialize auth state from localStorage
    useEffect(() => {
        const initAuth = async () => {
            try {
                const storedUser = authService.getStoredUser();
                const isAuth = authService.isAuthenticated();

                if (isAuth && storedUser) {
                    // Verify token is still valid by fetching current user
                    try {
                        const currentUser = await authService.getCurrentUser();
                        setUser(currentUser);
                        // Update stored user
                        localStorage.setItem(
                            "user",
                            JSON.stringify(currentUser)
                        );
                    } catch (error) {
                        // Token invalid, clear everything
                        authService.logout();
                        setUser(null);
                    }
                } else {
                    setUser(null);
                }
            } catch (error) {
                console.error("Auth initialization error:", error);
                setUser(null);
            } finally {
                setIsLoading(false);
            }
        };

        initAuth();
    }, []);

    const login = async (credentials: LoginRequest) => {
        try {
            const response = await authService.login(credentials);
            authService.storeAuthData(response);
            setUser(response.user);
        } catch (error) {
            throw new Error(handleApiError(error));
        }
    };

    const register = async (data: RegisterRequest) => {
        try {
            const response = await authService.register(data);
            authService.storeAuthData({
                access_token: response.access_token,
                refresh_token: response.refresh_token,
                token_type: response.token_type,
                user: response.user,
            });
            setUser(response.user);
        } catch (error) {
            throw new Error(handleApiError(error));
        }
    };

    const logout = async () => {
        try {
            await authService.logout();
        } catch (error) {
            console.error("Logout error:", error);
        } finally {
            setUser(null);
        }
    };

    const refreshUser = async () => {
        try {
            const currentUser = await authService.getCurrentUser();
            setUser(currentUser);
            localStorage.setItem("user", JSON.stringify(currentUser));
        } catch (error) {
            console.error("Refresh user error:", error);
        }
    };

    return (
        <AuthContext.Provider
            value={{
                user,
                isAuthenticated: !!user,
                isLoading,
                login,
                register,
                logout,
                refreshUser,
            }}
        >
            {children}
        </AuthContext.Provider>
    );
};
