import React, { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../../hooks/useAuth";
import { useNotification } from "../../hooks/useNotification";
import { Input } from "../../components/common/Input";
import { Button } from "../../components/common/Button";
import { Shield } from "lucide-react";

const LoginPage: React.FC = () => {
    const navigate = useNavigate();
    const { login, user } = useAuth();
    const { showNotification } = useNotification();

    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [loading, setLoading] = useState(false);

    // Redirect if already logged in
    useEffect(() => {
        if (user) {
            navigate(user.is_admin ? "/admin/dashboard" : "/dashboard");
        }
    }, [user, navigate]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if (!email || !password) {
            showNotification("error", "Please fill in all fields");
            return;
        }

        setLoading(true);

        try {
            await login({ email, password });
            showNotification("success", "Login successful!");
            // Navigation handled by useEffect above
        } catch (error: any) {
            showNotification(
                "error",
                error.response?.data?.detail ||
                    "Login failed. Please check your credentials."
            );
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-background px-4">
            <div className="w-full max-w-md">
                {/* Logo/Brand */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-linear-to-br from-accent to-active mb-4">
                        <Shield className="w-8 h-8 text-white" />
                    </div>
                    <h1 className="text-3xl font-bold my-gradient">
                        Insurance Fraud Detection
                    </h1>
                    <p className="text-muted mt-2">
                        Intelligent Risk-Based Authentication
                    </p>
                </div>

                {/* Login Card */}
                <div className="card">
                    <h2 className="text-2xl font-semibold text-primary mb-6">
                        Sign In
                    </h2>

                    <form onSubmit={handleSubmit} className="space-y-4">
                        <Input
                            label="Email Address"
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="your.email@example.com"
                            // icon={<Mail className="w-5 h-5" />}
                            disabled={loading}
                            required
                        />

                        <Input
                            label="Password"
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder="Enter your password"
                            // icon={<Lock className="w-5 h-5" />}
                            disabled={loading}
                            required
                        />

                        <Button
                            type="submit"
                            variant="primary"
                            fullWidth
                            isLoading={loading}
                            className="mt-6"
                        >
                            Sign In
                        </Button>
                    </form>

                    <div className="mt-6 text-center">
                        <p className="text-muted">
                            Don't have an account?{" "}
                            <Link
                                to="/register"
                                className="text-accent hover:text-active transition-colors font-medium"
                            >
                                Create Account
                            </Link>
                        </p>
                    </div>
                </div>

                {/* Demo Info */}
                <div className="mt-6 p-4 bg-surface/50 border border-aux rounded-lg">
                    <p className="text-sm text-muted text-center">
                        Demo System • Intelligent Risk Assessment & Fraud
                        Detection
                    </p>
                </div>
            </div>
        </div>
    );
};

export default LoginPage;
