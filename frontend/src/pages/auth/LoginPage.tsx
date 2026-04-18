import React, { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../../hooks/useAuth";
import { useNotification } from "../../hooks/useNotification";
import { Input } from "../../components/common/Input";
import { Button } from "../../components/common/Button";
import { Shield, Mail, Lock, ArrowRight, CheckCircle } from "lucide-react";
import loginBanner from "../../assets/login-banner.png";

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
        <div className="min-h-screen flex bg-background bg-mesh overflow-hidden">
            {/* Left Side: Visual/Branding (Hidden on mobile) */}
            <div className="hidden lg:flex lg:w-1/2 relative p-12 flex-col justify-between">
                <div className="absolute inset-0 z-0">
                    <img 
                        src={loginBanner} 
                        alt="Security Banner" 
                        className="w-full h-full object-cover opacity-60 mix-blend-luminosity grayscale hover:grayscale-0 transition-all duration-1000"
                    />
                    <div className="absolute inset-0 bg-linear-to-t from-background via-transparent to-transparent"></div>
                </div>

                <div className="relative z-10">
                    <div className="flex items-center gap-3 mb-12">
                        <div className="w-12 h-12 rounded-xl bg-accent/20 backdrop-blur-md flex items-center justify-center border border-accent/30">
                            <Shield className="w-7 h-7 text-accent" />
                        </div>
                        <span className="text-2xl font-bold tracking-tight text-white">Insurance <span className="text-accent">Portal</span></span>
                    </div>

                    <div className="max-w-md">
                        <h2 className="text-5xl font-extrabold text-white leading-tight mb-6">
                            Secure Claims <br />
                            <span className="my-gradient">Intelligently.</span>
                        </h2>
                        <ul className="space-y-4">
                            {[
                                "AI-Powered Risk Assessment",
                                "Zero-Trust Authentication",
                                "Real-time Fraud Prevention"
                            ].map((item, i) => (
                                <li key={i} className="flex items-center gap-3 text-lg text-white/80">
                                    <CheckCircle className="w-5 h-5 text-accent" />
                                    {item}
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>

                <div className="relative z-10">
                    <p className="text-white/50 text-sm">
                        &copy; 2026 Insurance Fraud Protection. All rights reserved.
                    </p>
                </div>
            </div>

            {/* Right Side: Login Form */}
            <div className="w-full lg:w-1/2 flex items-center justify-center p-6 sm:p-12 relative z-10 auth-container">
                <div className="w-full max-w-md">
                    <div className="lg:hidden text-center mb-8">
                        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-accent/10 mb-4 border border-accent/20">
                            <Shield className="w-8 h-8 text-accent" />
                        </div>
                        <h1 className="text-3xl font-bold text-white">Welcome Back</h1>
                    </div>

                    <div className="card-glass p-8 sm:p-10">
                        <div className="mb-8">
                            <h2 className="text-3xl font-bold text-white mb-2">Sign In</h2>
                            <p className="text-muted">Enter your details to access your dashboard</p>
                        </div>

                        <form onSubmit={handleSubmit} className="space-y-6">
                            <Input
                                label="Email Address"
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="name@example.com"
                                icon={<Mail className="w-5 h-5" />}
                                disabled={loading}
                                required
                                className="bg-transparent!"
                            />

                            <div>
                                <div className="flex justify-between mb-1">
                                    <label className="text-sm font-medium text-white/70">Password</label>
                                    <Link to="/forgot-password" title="Forgot Password Feature Coming Soon" className="text-sm text-accent hover:underline">
                                        Forgot?
                                    </Link>
                                </div>
                                <Input
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="••••••••"
                                    icon={<Lock className="w-5 h-5" />}
                                    disabled={loading}
                                    required
                                    className="bg-transparent!"
                                />
                            </div>

                            <Button
                                type="submit"
                                variant="primary"
                                fullWidth
                                isLoading={loading}
                                className="h-12 text-lg font-semibold group"
                            >
                                <span className="flex items-center justify-center gap-2">
                                    Access Dashboard
                                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                                </span>
                            </Button>
                        </form>

                        <div className="mt-8 text-center">
                            <p className="text-muted">
                                New to the platform?{" "}
                                <Link
                                    to="/register"
                                    className="text-accent hover:text-active transition-colors font-semibold"
                                >
                                    Create an account
                                </Link>
                            </p>
                        </div>
                    </div>

                    {/* Footer Info */}
                    <div className="mt-8 text-center text-xs text-muted flex items-center justify-center gap-4">
                        <span className="px-3 py-1 bg-surface/30 rounded-full border border-aux/50 overflow-hidden">
                            SECURE SESSION
                        </span>
                        <span>AES-256 ENCRYPTED</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LoginPage;
