import React, { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../../hooks/useAuth";
import { useNotification } from "../../hooks/useNotification";
import { Input } from "../../components/common/Input";
import { Button } from "../../components/common/Button";
import { Select } from "../../components/common/Select";
import {
    Shield,
    Mail,
    Lock,
    User,
    Phone,
    MapPin,
    CreditCard,
} from "lucide-react";

const RegisterPage: React.FC = () => {
    const navigate = useNavigate();
    const { register, user } = useAuth();
    const { showNotification } = useNotification();

    const [formData, setFormData] = useState({
        email: "",
        password: "",
        confirmPassword: "",
        first_name: "",
        last_name: "",
        phone_number: "",
        address: "",
        city: "",
        state: "",
        country: "Nigeria",
        postal_code: "",
        policy_type: "auto",
    });

    const [loading, setLoading] = useState(false);

    // Redirect if already logged in
    useEffect(() => {
        if (user) {
            navigate(user.is_admin ? "/admin/dashboard" : "/dashboard");
        }
    }, [user, navigate]);

    const handleChange = (field: string, value: string) => {
        setFormData((prev) => ({ ...prev, [field]: value }));
    };

    const validateForm = (): boolean => {
        if (
            !formData.email ||
            !formData.password ||
            !formData.first_name ||
            !formData.last_name
        ) {
            showNotification("error", "Please fill in all required fields");
            return false;
        }

        if (formData.password !== formData.confirmPassword) {
            showNotification("error", "Passwords do not match");
            return false;
        }

        if (formData.password.length < 8) {
            showNotification("error", "Password must be at least 8 characters");
            return false;
        }

        return true;
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if (!validateForm()) return;

        setLoading(true);

        try {
            const { confirmPassword, ...registrationData } = formData;
            await register(registrationData);
            showNotification(
                "success",
                "Registration successful! Welcome aboard."
            );
            // Navigation handled by useEffect
        } catch (error: any) {
            showNotification(
                error.response?.data?.detail ||
                    "Registration failed. Please try again.",
                "error"
            );
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-background px-4 py-12">
            <div className="w-full max-w-2xl">
                {/* Logo/Brand */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-linear-to-br from-accent to-active mb-4">
                        <Shield className="w-8 h-8 text-white" />
                    </div>
                    <h1 className="text-3xl font-bold my-gradient">
                        Create Your Account
                    </h1>
                    <p className="text-muted mt-2">
                        Join our intelligent insurance platform
                    </p>
                </div>

                {/* Registration Card */}
                <div className="card">
                    <form onSubmit={handleSubmit} className="space-y-6">
                        {/* Personal Information */}
                        <div>
                            <h3 className="text-lg font-semibold text-primary mb-4">
                                Personal Information
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <Input
                                    label="First Name"
                                    type="text"
                                    value={formData.first_name}
                                    onChange={(e) =>
                                        handleChange(
                                            "first_name",
                                            e.target.value
                                        )
                                    }
                                    placeholder="John"
                                    icon={<User className="w-5 h-5" />}
                                    disabled={loading}
                                    required
                                />
                                <Input
                                    label="Last Name"
                                    type="text"
                                    value={formData.last_name}
                                    onChange={(e) =>
                                        handleChange(
                                            "last_name",
                                            e.target.value
                                        )
                                    }
                                    placeholder="Doe"
                                    icon={<User className="w-5 h-5" />}
                                    disabled={loading}
                                    required
                                />
                            </div>
                        </div>

                        {/* Contact Information */}
                        <div>
                            <h3 className="text-lg font-semibold text-primary mb-4">
                                Contact Information
                            </h3>
                            <div className="space-y-4">
                                <Input
                                    label="Email Address"
                                    type="email"
                                    value={formData.email}
                                    onChange={(e) =>
                                        handleChange("email", e.target.value)
                                    }
                                    placeholder="your.email@example.com"
                                    icon={<Mail className="w-5 h-5" />}
                                    disabled={loading}
                                    required
                                />
                                <Input
                                    label="Phone Number"
                                    type="tel"
                                    value={formData.phone_number}
                                    onChange={(e) =>
                                        handleChange(
                                            "phone_number",
                                            e.target.value
                                        )
                                    }
                                    placeholder="+234 800 000 0000"
                                    icon={<Phone className="w-5 h-5" />}
                                    disabled={loading}
                                />
                            </div>
                        </div>

                        {/* Address Information */}
                        <div>
                            <h3 className="text-lg font-semibold text-primary mb-4">
                                Address
                            </h3>
                            <div className="space-y-4">
                                <Input
                                    label="Street Address"
                                    type="text"
                                    value={formData.address}
                                    onChange={(e) =>
                                        handleChange("address", e.target.value)
                                    }
                                    placeholder="123 Main Street"
                                    icon={<MapPin className="w-5 h-5" />}
                                    disabled={loading}
                                />
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <Input
                                        label="City"
                                        type="text"
                                        value={formData.city}
                                        onChange={(e) =>
                                            handleChange("city", e.target.value)
                                        }
                                        placeholder="Lagos"
                                        disabled={loading}
                                    />
                                    <Input
                                        label="State"
                                        type="text"
                                        value={formData.state}
                                        onChange={(e) =>
                                            handleChange(
                                                "state",
                                                e.target.value
                                            )
                                        }
                                        placeholder="Lagos"
                                        disabled={loading}
                                    />
                                </div>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <Input
                                        label="Country"
                                        type="text"
                                        value={formData.country}
                                        onChange={(e) =>
                                            handleChange(
                                                "country",
                                                e.target.value
                                            )
                                        }
                                        placeholder="Nigeria"
                                        disabled={loading}
                                    />
                                    <Input
                                        label="Postal Code"
                                        type="text"
                                        value={formData.postal_code}
                                        onChange={(e) =>
                                            handleChange(
                                                "postal_code",
                                                e.target.value
                                            )
                                        }
                                        placeholder="100001"
                                        disabled={loading}
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Policy Information */}
                        <div>
                            <h3 className="text-lg font-semibold text-primary mb-4">
                                Insurance Policy
                            </h3>
                            <Select
                                label="Policy Type"
                                value={formData.policy_type}
                                onChange={(e) =>
                                    handleChange("policy_type", e.target.value)
                                }
                                icon={<CreditCard className="w-5 h-5" />}
                                disabled={loading}
                                options={[
                                    { value: "auto", label: "Auto Insurance" },
                                    {
                                        value: "health",
                                        label: "Health Insurance",
                                    },
                                    { value: "home", label: "Home Insurance" },
                                    { value: "life", label: "Life Insurance" },
                                ]}
                            >
                                <option value="auto">Auto Insurance</option>
                                <option value="health">Health Insurance</option>
                                <option value="home">Home Insurance</option>
                                <option value="life">Life Insurance</option>
                            </Select>
                        </div>

                        {/* Security */}
                        <div>
                            <h3 className="text-lg font-semibold text-primary mb-4">
                                Security
                            </h3>
                            <div className="space-y-4">
                                <Input
                                    label="Password"
                                    type="password"
                                    value={formData.password}
                                    onChange={(e) =>
                                        handleChange("password", e.target.value)
                                    }
                                    placeholder="Min. 8 characters"
                                    icon={<Lock className="w-5 h-5" />}
                                    disabled={loading}
                                    required
                                />
                                <Input
                                    label="Confirm Password"
                                    type="password"
                                    value={formData.confirmPassword}
                                    onChange={(e) =>
                                        handleChange(
                                            "confirmPassword",
                                            e.target.value
                                        )
                                    }
                                    placeholder="Re-enter password"
                                    icon={<Lock className="w-5 h-5" />}
                                    disabled={loading}
                                    required
                                />
                            </div>
                        </div>

                        <Button
                            type="submit"
                            variant="primary"
                            fullWidth
                            isLoading={loading}
                            className="mt-6"
                        >
                            Create Account
                        </Button>
                    </form>

                    <div className="mt-6 text-center">
                        <p className="text-muted">
                            Already have an account?{" "}
                            <Link
                                to="/login"
                                className="text-accent hover:text-active transition-colors font-medium"
                            >
                                Sign In
                            </Link>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default RegisterPage;
