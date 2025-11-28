import type { User } from "./user.types";

export interface LoginRequest {
    email: string;
    password: string;
}

export interface RegisterRequest {
    email: string;
    password: string;
    first_name: string;
    last_name: string;
    phone_number?: string;
    city?: string;
    state?: string;
    country?: string;
    policy_type?: string;
}

export interface AuthResponse {
    access_token: string;
    refresh_token: string;
    token_type: string;
    user: User;
}

export interface LoginResponse extends AuthResponse {}

export interface RegisterResponse {
    message: string;
    user: User;
    access_token: string;
    refresh_token: string;
    token_type: string;
}
