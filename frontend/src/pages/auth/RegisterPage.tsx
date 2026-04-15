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
	ArrowRight,
	ArrowLeft,
	CheckCircle,
} from "lucide-react";
import registerBanner from "../../assets/register-banner.png";

const RegisterPage: React.FC = () => {
	const navigate = useNavigate();
	const { register, user } = useAuth();
	const { showNotification } = useNotification();

	const [step, setStep] = useState(1);
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

	const validateStep = (currentStep: number): boolean => {
		if (currentStep === 1) {
			if (!formData.first_name || !formData.last_name) {
				showNotification("error", "Please enter your full name");
				return false;
			}
		} else if (currentStep === 2) {
			if (!formData.email) {
				showNotification("error", "Please enter your email address");
				return false;
			}
		} else if (currentStep === 4) {
			if (!formData.password || !formData.confirmPassword) {
				showNotification("error", "Please set a password");
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
		}
		return true;
	};

	const nextStep = () => {
		if (validateStep(step)) {
			setStep((prev) => prev + 1);
		}
	};

	const prevStep = () => {
		setStep((prev) => prev - 1);
	};

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();

		if (!validateStep(4)) return;

		setLoading(true);

		try {
			const { confirmPassword, ...registrationData } = formData;
			await register(registrationData);
			showNotification("success", "Registration successful! Welcome aboard.");
		} catch (error: any) {
			showNotification(
				"error",
				error.response?.data?.detail ||
					"Registration failed. Please try again.",
			);
		} finally {
			setLoading(false);
		}
	};

	const steps = [
		{ id: 1, title: "Identity", icon: <User className="w-5 h-5" /> },
		{ id: 2, title: "Contact", icon: <Mail className="w-5 h-5" /> },
		{ id: 3, title: "Location", icon: <MapPin className="w-5 h-5" /> },
		{ id: 4, title: "Policy", icon: <Shield className="w-5 h-5" /> },
	];

	return (
		<div className="min-h-screen flex bg-background bg-mesh overflow-hidden">
			{/* Left Side: Visual/Branding */}
			<div className="hidden lg:flex lg:w-2/5 relative p-12 flex-col justify-between">
				<div className="absolute inset-0 z-0">
					<img
						src={registerBanner}
						alt="Onboarding Banner"
						className="w-full h-full object-cover opacity-60 mix-blend-overlay"
					/>
					<div className="absolute inset-0 bg-linear-to-r from-background via-transparent to-transparent"></div>
				</div>

				<div className="relative z-10">
					<Link to="/" className="flex items-center gap-3 mb-12">
						<div className="w-10 h-10 rounded-lg bg-accent/20 backdrop-blur-md flex items-center justify-center border border-accent/30">
							<Shield className="w-6 h-6 text-accent" />
						</div>
						<span className="text-xl font-bold tracking-tight text-white">
							InsureGuard <span className="text-accent">AI</span>
						</span>
					</Link>

					<div className="max-w-xs">
						<h2 className="text-4xl font-extrabold text-white leading-tight mb-6">
							Join the <br />
							<span className="my-gradient">Future</span> of <br />
							Insurance.
						</h2>
						<p className="text-white/60 text-lg">
							Get started in minutes with our streamlined registration process.
						</p>
					</div>
				</div>

				<div className="relative z-10 flex items-center gap-4">
					<div className="flex -space-x-3">
						{[1, 2, 3, 4].map((i) => (
							<div
								key={i}
								className="w-10 h-10 rounded-full border-2 border-background bg-surface flex items-center justify-center overflow-hidden"
							>
								<img src={`https://i.pravatar.cc/100?u=${i}`} alt="User" />
							</div>
						))}
					</div>
					<p className="text-white/50 text-sm font-medium">
						Joined by +10k users this month
					</p>
				</div>
			</div>

			{/* Right Side: Step-by-Step Form */}
			<div className="w-full lg:w-3/5 flex items-center justify-center p-6 sm:p-12 relative z-10 auth-container">
				<div className="w-full max-w-xl">
					{/* Progress Indicator */}
					<div className="mb-12">
						<div className="flex items-center justify-between mb-4">
							{steps.map((s) => (
								<div key={s.id} className="flex flex-col items-center gap-2">
									<div
										className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300 ${
											step >= s.id
												? "bg-accent border-accent text-white"
												: "bg-surface border-aux text-muted"
										}`}
									>
										{step > s.id ? <CheckCircle className="w-6 h-6" /> : s.icon}
									</div>
									<span
										className={`text-xs font-semibold ${step >= s.id ? "text-accent" : "text-muted"}`}
									>
										{s.title}
									</span>
								</div>
							))}
						</div>
						<div className="h-1.5 w-full bg-surface rounded-full overflow-hidden">
							<div
								className="h-full bg-accent transition-all duration-500 ease-out"
								style={{ width: `${((step - 1) / (steps.length - 1)) * 100}%` }}
							></div>
						</div>
					</div>

					<div className="card-glass p-8 sm:p-10">
						<form onSubmit={handleSubmit} className="space-y-6">
							{/* Step 1: Personal Info */}
							{step === 1 && (
								<div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-500">
									<div>
										<h3 className="text-2xl font-bold text-white mb-2">
											Personal Identity
										</h3>
										<p className="text-muted mb-6">
											Let's start with your legal name as it appears on your ID.
										</p>
									</div>
									<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
										<Input
											label="First Name"
											value={formData.first_name}
											onChange={(e) =>
												handleChange("first_name", e.target.value)
											}
											placeholder="John"
											icon={<User className="w-5 h-5" />}
											required
										/>
										<Input
											label="Last Name"
											value={formData.last_name}
											onChange={(e) =>
												handleChange("last_name", e.target.value)
											}
											placeholder="Doe"
											icon={<User className="w-5 h-5" />}
											required
										/>
									</div>
								</div>
							)}

							{/* Step 2: Contact Info */}
							{step === 2 && (
								<div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-500">
									<div>
										<h3 className="text-2xl font-bold text-white mb-2">
											Contact Details
										</h3>
										<p className="text-muted mb-6">
											How should we reach you for policy updates?
										</p>
									</div>
									<Input
										label="Email Address"
										type="email"
										value={formData.email}
										onChange={(e) => handleChange("email", e.target.value)}
										placeholder="john.doe@example.com"
										icon={<Mail className="w-5 h-5" />}
										required
									/>
									<Input
										label="Phone Number"
										type="tel"
										value={formData.phone_number}
										onChange={(e) =>
											handleChange("phone_number", e.target.value)
										}
										placeholder="+234 800 000 0000"
										icon={<Phone className="w-5 h-5" />}
									/>
								</div>
							)}

							{/* Step 3: Address */}
							{step === 3 && (
								<div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-500">
									<div>
										<h3 className="text-2xl font-bold text-white mb-2">
											Your Address
										</h3>
										<p className="text-muted mb-6">
											We use this to determine regional policy rates.
										</p>
									</div>
									<Input
										label="Street Address"
										value={formData.address}
										onChange={(e) => handleChange("address", e.target.value)}
										placeholder="123 Main Street"
										icon={<MapPin className="w-5 h-5" />}
									/>
									<div className="grid grid-cols-2 gap-4">
										<Input
											label="City"
											value={formData.city}
											onChange={(e) => handleChange("city", e.target.value)}
											placeholder="Lagos"
										/>
										<Input
											label="State"
											value={formData.state}
											onChange={(e) => handleChange("state", e.target.value)}
											placeholder="Lagos"
										/>
									</div>
									<div className="grid grid-cols-2 gap-4">
										<Input
											label="Country"
											value={formData.country}
											onChange={(e) => handleChange("country", e.target.value)}
											placeholder="Nigeria"
										/>
										<Input
											label="Postal Code"
											value={formData.postal_code}
											onChange={(e) =>
												handleChange("postal_code", e.target.value)
											}
											placeholder="100001"
										/>
									</div>
								</div>
							)}

							{/* Step 4: Policy & Security */}
							{step === 4 && (
								<div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-500">
									<div>
										<h3 className="text-2xl font-bold text-white mb-2">
											Final Security
										</h3>
										<p className="text-muted mb-6">
											Choose your policy type and secure your account.
										</p>
									</div>
									<Select
										label="Primary Policy Type"
										value={formData.policy_type}
										onChange={(e) =>
											handleChange("policy_type", e.target.value)
										}
										icon={<CreditCard className="w-5 h-5" />}
										options={[
											{ value: "auto", label: "Auto Insurance" },
											{ value: "health", label: "Health Insurance" },
											{ value: "home", label: "Home Insurance" },
											{ value: "life", label: "Life Insurance" },
										]}
									/>
									<div className="space-y-4">
										<Input
											label="Create Password"
											type="password"
											value={formData.password}
											onChange={(e) => handleChange("password", e.target.value)}
											placeholder="Min. 8 characters"
											icon={<Lock className="w-5 h-5" />}
											required
										/>
										<Input
											label="Confirm Password"
											type="password"
											value={formData.confirmPassword}
											onChange={(e) =>
												handleChange("confirmPassword", e.target.value)
											}
											placeholder="Re-enter password"
											icon={<Lock className="w-5 h-5" />}
											required
										/>
									</div>
								</div>
							)}

							{/* Navigation Buttons */}
							<div className="flex gap-4 mt-8">
								{step > 1 && (
									<Button
										type="button"
										variant="secondary"
										onClick={prevStep}
										className="flex-1 h-12"
										disabled={loading}
									>
										<span className="flex items-center justify-center gap-2">
											<ArrowLeft className="w-5 h-5" />
											Back
										</span>
									</Button>
								)}

								{step < 4 ? (
									<Button
										type="button"
										variant="primary"
										onClick={nextStep}
										className="flex-2 h-12 font-semibold group"
									>
										<span className="flex items-center justify-center gap-2">
											Continue
											<ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
										</span>
									</Button>
								) : (
									<Button
										type="submit"
										variant="primary"
										fullWidth
										isLoading={loading}
										className="flex-2 h-12 font-bold"
									>
										Complete Enrollment
									</Button>
								)}
							</div>
						</form>

						<div className="mt-8 text-center">
							<p className="text-muted">
								Already have an account?{" "}
								<Link
									to="/login"
									className="text-accent hover:text-active transition-colors font-semibold"
								>
									Sign In
								</Link>
							</p>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
};

export default RegisterPage;
