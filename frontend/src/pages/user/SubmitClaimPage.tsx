import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useNotification } from "../../hooks/useNotification";
import { useSessionTrackingContext } from "../../contexts/SessionTrackingContext";
import { claimService } from "../../services/claimService";
import { Card } from "../../components/common/Card";
import { Input } from "../../components/common/Input";
import { Select } from "../../components/common/Select";
import { Textarea } from "../../components/common/Textarea";
import { Button } from "../../components/common/Button";
import { Modal } from "../../components/common/Modal";
import Navbar from "../../components/layout/Navbar";
import OTPVerification from "../../components/mfa/OTPVerification";
import BiometricVerification from "../../components/mfa/BiometricVerification";
import {
	FileText,
	Calendar,
	DollarSign,
	AlertCircle,
	Shield,
	CheckCircle,
	Settings,
	MapPin,
} from "lucide-react";
import type {
	ClaimSubmission,
	ClaimSubmissionResponse,
} from "../../types/claim.types";

const SubmitClaimPage: React.FC = () => {
	const navigate = useNavigate();
	const { showNotification } = useNotification();
	const { getSessionDuration, sessionData } = useSessionTrackingContext();

	const [formData, setFormData] = useState<ClaimSubmission>({
		claim_type: "accident",
		claim_amount: 0,
		incident_date: "",
		claim_description: "",
		supporting_documents_count: 0,
		session_duration: 0,
		pages_visited: 0,
		form_fill_time: 0,
	});

	const formStartTimeRef = useRef<number>(Date.now());
	const [loading, setLoading] = useState(false);
	const [showMFAModal, setShowMFAModal] = useState(false);
	const [mfaMethod, setMfaMethod] = useState<"otp" | "biometric" | null>(null);
	const [claimResponse, setClaimResponse] =
		useState<ClaimSubmissionResponse | null>(null);

	const handleChange = (
		field: keyof ClaimSubmission,
		value: string | number | boolean,
	) => {
		setFormData((prev) => ({ ...prev, [field]: value }));
	};

	const validateForm = (): boolean => {
		if (
			!formData.claim_type ||
			formData.claim_amount <= 0 ||
			!formData.incident_date
		) {
			showNotification("error", "Please fill in all required fields");
			return false;
		}

		if (!formData.claim_description || formData.claim_description.length < 10) {
			showNotification(
				"error",
				"Please provide a detailed description (min 10 characters)",
			);
			return false;
		}

		const incidentDate = new Date(formData.incident_date);
		const today = new Date();
		if (incidentDate > today) {
			showNotification("error", "Incident date cannot be in the future");
			return false;
		}

		return true;
	};

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();

		if (!validateForm()) return;

		setLoading(true);

		try {
			// Calculate form fill time in seconds
			const formFillTime = Math.floor(
				(Date.now() - formStartTimeRef.current) / 1000,
			);

			// Prepare claim data with timing information
			const claimDataWithTiming = {
				...formData,
				form_fill_time: formFillTime,
				session_duration: getSessionDuration(),
				pages_visited: sessionData.pages_visited,
			};

			console.log("CLAIM DATA:", claimDataWithTiming);
			const response = await claimService.submitClaim(claimDataWithTiming);
			setClaimResponse(response);
			console.log("RESPONSE:", response);

			if (response.requires_mfa) {
				// MFA required - show modal
				setMfaMethod(response.mfa_method as "otp" | "biometric");
				setShowMFAModal(true);
				showNotification(
					"warning",
					`Additional verification required: ${response.mfa_method?.toUpperCase()}`,
				);
			} else {
				// Auto-approved
				showNotification(
					"success",
					response.message || "Claim submitted successfully!",
				);
				setTimeout(() => {
					navigate(`/claims/${response.claim.claim_id}`);
				}, 2000);
			}
		} catch (error: any) {
			showNotification(
				"error",
				error.response?.data?.detail ||
					"Failed to submit claim. Please try again.",
			);
		} finally {
			setLoading(false);
		}
	};

	const handleMFASuccess = () => {
		setShowMFAModal(false);
		showNotification("success", "Claim verified and approved!");
		setTimeout(() => {
			if (claimResponse?.claim?.claim_id) {
				navigate(`/claims/${claimResponse.claim.claim_id}`);
			} else {
				navigate("/claims/history");
			}
		}, 1500);
	};

	const handleMFACancel = () => {
		setShowMFAModal(false);
		showNotification(
			"info",
			"Claim submitted but pending MFA verification. You can complete it from your claims page.",
		);
		navigate("/claims/history");
	};

	// Update session data when tracking changes
	useEffect(() => {
		setFormData((prev) => ({
			...prev,
			session_duration: getSessionDuration(),
			pages_visited: sessionData.pages_visited,
		}));
	}, [getSessionDuration, sessionData.pages_visited]);

	const formatCurrency = (amount: number) => {
		return new Intl.NumberFormat("en-NG", {
			style: "currency",
			currency: "NGN",
		}).format(amount);
	};

	return (
		<div className="min-h-screen bg-background bg-mesh">
			<Navbar />

			<div className="max-w-7xl mx-auto px-4 pt-32 pb-12 relative z-10">
				{/* Header */}
				<div className="mb-12 animate-in fade-in slide-in-from-top-4 duration-700">
					<h1 className="text-4xl font-extrabold text-white mb-3">
						Claim Submission <span className="my-gradient">Portal</span>
					</h1>
					<p className="text-white/50 text-lg max-w-2xl">
						Securely submit your insurance claim. Our neural assessment system
						analyzes environmental and behavioral metadata in real-time.
					</p>
				</div>

				<div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
					{/* Main Form Column */}
					<div className="lg:col-span-2 space-y-8">
						{/* Security Notice */}
						<Card
							variant="glass"
							className="border-accent/30 bg-accent/5! relative overflow-hidden group"
						>
							<div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
								<Shield className="w-24 h-24 text-accent -mr-8 -mt-8 rotate-12" />
							</div>
							<div className="flex items-start gap-6 relative z-10">
								<div className="p-3 rounded-2xl bg-accent/20 border border-accent/30 shrink-0">
									<Shield className="w-8 h-8 text-accent" />
								</div>
								<div>
									<h3 className="text-xl font-bold text-white mb-2">
										Neural Risk Assessment Active
									</h3>
									<p className="text-white/60 text-lg leading-relaxed">
										Your claim is being analyzed in real-time for behavioral
										risk factors. High-entropy claims may trigger additional{" "}
										<span className="text-accent font-bold">MFA protocols</span>
										.
									</p>
								</div>
							</div>
						</Card>

						{/* Claim Form */}
						<Card variant="glass" className="border-white/10 shadow-2xl">
							<form onSubmit={handleSubmit} className="space-y-8">
								<div className="grid grid-cols-1 md:grid-cols-2 gap-8">
									{/* Claim Type */}
									<Select
										label="Claim Type"
										value={formData.claim_type}
										onChange={(e) => handleChange("claim_type", e.target.value)}
										icon={<FileText className="w-5 h-5" />}
										options={[
											{ value: "accident", label: "Accident" },
											{ value: "theft", label: "Theft" },
											{ value: "medical", label: "Medical" },
											{
												value: "property_damage",
												label: "Propery Damage",
											},
											{ value: "other", label: "Other" },
										]}
										disabled={loading}
										required
									/>

									{/* Incident Date */}
									<Input
										label="Incident Date"
										type="date"
										value={formData.incident_date}
										onChange={(e) =>
											handleChange("incident_date", e.target.value)
										}
										icon={<Calendar className="w-5 h-5" />}
										disabled={loading}
										required
										max={new Date().toISOString().split("T")[0]}
									/>
								</div>

								{/* Claim Amount */}
								<Input
									label="Estimated Claim Amount"
									type="number"
									value={formData.claim_amount || ""}
									onChange={(e) =>
										handleChange(
											"claim_amount",
											parseFloat(e.target.value) || 0,
										)
									}
									placeholder="0.00"
									icon={<DollarSign className="w-5 h-5" />}
									disabled={loading}
									required
									min="0"
									step="0.01"
									helperText={
										formData.claim_amount > 0
											? `Verified Value: ${formatCurrency(formData.claim_amount)}`
											: "Enter the estimated loss in Naira (NGN)"
									}
								/>

								{/* Description */}
								<Textarea
									label="Incident Narrative"
									value={formData.claim_description}
									onChange={(e) =>
										handleChange("claim_description", e.target.value)
									}
									placeholder="Provide a detailed chronological narrative of the incident..."
									disabled={loading}
									required
									rows={8}
									className="bg-background/50!"
									helperText="Detailed descriptions reduce risk scores and accelerate processing."
								/>

								{/* Supporting Documents Count */}
								<Input
									label="Supporting Metadata (Documents)"
									type="number"
									value={formData.supporting_documents_count}
									onChange={(e) =>
										handleChange(
											"supporting_documents_count",
											parseInt(e.target.value) || 0,
										)
									}
									placeholder="0"
									icon={<FileText className="w-5 h-5" />}
									disabled={loading}
									min="0"
									max="10"
									helperText="Total number of digital evidence files prepared for submission (0-10)"
								/>

								{/* Submit Button */}
								<div className="flex gap-4 pt-8">
									<Button
										type="button"
										variant="secondary"
										onClick={() => navigate("/dashboard")}
										disabled={loading}
										className="px-8 border-white/10 hover:bg-white/5"
									>
										Discard
									</Button>
									<Button
										type="submit"
										variant="primary"
										isLoading={loading}
										icon={<CheckCircle className="w-5 h-5" />}
										className="flex-1 h-14 text-lg font-bold shadow-lg shadow-accent/20"
									>
										Finalize Submission
									</Button>
								</div>
							</form>
						</Card>
					</div>

					{/* Sidebar Column: Simulation Environment */}
					<div className="lg:sticky lg:top-32 space-y-6">
						<Card
							variant="glass"
							className="border-accent/10 bg-black/40! overflow-hidden shadow-2xl"
						>
							<div className="p-6 border-b border-white/5 bg-accent/5">
								<div className="flex items-center gap-3 text-accent font-black uppercase tracking-widest text-xs">
									<Settings className="w-5 h-5" />
									Environmental Overrides
								</div>
								<h3 className="text-xl font-bold text-white mt-3">
									Simulation Lab
								</h3>
								<p className="text-white/40 text-sm mt-2">
									Manipulate behavioral and environmental metadata to test the
									RBA/MFA logic.
								</p>
							</div>

							<div className="p-6 space-y-8">
								{/* Trusted Device */}
								<div className="space-y-3">
									<div className="flex items-center justify-between">
										<label className="text-sm font-bold text-white/60 tracking-tight">
											Device Integrity
										</label>
										<span
											className={`text-[10px] font-black px-2 py-0.5 rounded uppercase ${formData.is_trusted_device ? "bg-success/20 text-success" : "bg-white/5 text-white/30"}`}
										>
											{formData.is_trusted_device ? "Verified" : "Unknown"}
										</span>
									</div>
									<div
										onClick={() =>
											handleChange(
												"is_trusted_device",
												!formData.is_trusted_device,
											)
										}
										className={`p-4 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
											formData.is_trusted_device
												? "bg-accent/10 border-accent/40 shadow-inner shadow-accent/10"
												: "bg-white/5 border-white/5 hover:bg-white/10"
										}`}
									>
										<div className="flex items-center gap-3">
											<Shield
												className={`w-5 h-5 ${formData.is_trusted_device ? "text-accent" : "text-white/20"}`}
											/>
											<span className="text-sm font-medium text-white italic">
												Trusted Device
											</span>
										</div>
										<div
											className={`w-10 h-5 rounded-full relative transition-colors ${formData.is_trusted_device ? "bg-accent" : "bg-white/10"}`}
										>
											<div
												className={`absolute top-1 w-3 h-3 rounded-full bg-white transition-all ${formData.is_trusted_device ? "right-1" : "left-1"}`}
											/>
										</div>
									</div>
								</div>

								{/* Device Trust Score */}
								<div className="space-y-4">
									<div className="flex justify-between items-center">
										<label className="text-sm font-bold text-white/60 tracking-tight">
											Trust Vector
										</label>
										<span className="text-lg font-black text-accent font-mono">
											{(formData.device_trust_score ?? 0.5).toFixed(1)}
										</span>
									</div>
									<input
										type="range"
										min="0"
										max="1"
										step="0.1"
										value={formData.device_trust_score ?? 0.5}
										onChange={(e) =>
											handleChange(
												"device_trust_score",
												parseFloat(e.target.value),
											)
										}
										className="w-full h-2 bg-white/5 rounded-lg appearance-none cursor-pointer accent-accent"
									/>
								</div>

								{/* Geolocation Anomaly */}
								<div className="space-y-3">
									<div className="flex items-center justify-between">
										<label className="text-sm font-bold text-white/60 tracking-tight">
											Location Integrity
										</label>
										<span
											className={`text-[10px] font-black px-2 py-0.5 rounded uppercase ${formData.is_geolocation_anomaly ? "bg-error/20 text-error animate-pulse" : "bg-success/20 text-success"}`}
										>
											{formData.is_geolocation_anomaly ? "Anomaly" : "Valid"}
										</span>
									</div>
									<div
										onClick={() =>
											handleChange(
												"is_geolocation_anomaly",
												!formData.is_geolocation_anomaly,
											)
										}
										className={`p-4 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
											formData.is_geolocation_anomaly
												? "bg-error/10 border-error/40"
												: "bg-white/5 border-white/5 hover:bg-white/10"
										}`}
									>
										<div className="flex items-center gap-3">
											<MapPin
												className={`w-5 h-5 ${formData.is_geolocation_anomaly ? "text-error" : "text-white/20"}`}
											/>
											<span className="text-sm font-medium text-white italic">
												Geo-Anomaly
											</span>
										</div>
									</div>
								</div>

								{/* Geolocation Distance */}
								<div className="space-y-2">
									<label className="text-sm font-bold text-white/60 tracking-tight">
										Distance (km)
									</label>
									<div className="relative group">
										<div className="absolute inset-y-0 left-4 flex items-center pointer-events-none">
											<AlertCircle className="w-5 h-5 text-white/20 group-focus-within:text-accent transition-colors" />
										</div>
										<input
											type="number"
											value={formData.geolocation_distance_km ?? 120}
											onChange={(e) =>
												handleChange(
													"geolocation_distance_km",
													parseFloat(e.target.value),
												)
											}
											className="w-full bg-white/5 border border-white/5 rounded-xl pl-12 pr-4 py-4 text-accent font-black text-xl focus:border-accent/40 focus:ring-0 transition-all"
										/>
									</div>
								</div>
							</div>
						</Card>

						<div className="p-6 rounded-2xl bg-white/5 border border-white/5 italic">
							<p className="text-xs text-white/40 leading-relaxed">
								Use these controls to simulate various risk scenarios. The
								backend ML model will weigh these factors alongside your
								behavioral telemetry to determine the security challenge level.
							</p>
						</div>
					</div>
				</div>
			</div>

			{/* MFA Modal */}
			{showMFAModal && claimResponse && (
				<Modal
					isOpen={showMFAModal}
					onClose={handleMFACancel}
					title="Neural Verification Challenge"
					size="md"
				>
					<div className="space-y-6">
						<div className="flex items-start gap-4 p-5 rounded-2xl bg-warning/10 border border-warning/20 animate-pulse-red">
							<AlertCircle className="w-6 h-6 text-warning shrink-0 mt-0.5" />
							<div className="text-sm">
								<p className="font-black text-white uppercase tracking-widest text-xs mb-2">
									High Probability Risk Detected
								</p>
								<p className="text-white/60 text-base leading-relaxed">
									Assessment Level:{" "}
									<span className="text-warning font-black">
										{claimResponse.risk_assessment.risk_level.toUpperCase()}
									</span>
									. Additional biometric or cryptographic verification is
									required to authorize this claim.
								</p>
							</div>
						</div>

						{mfaMethod === "otp" ? (
							<OTPVerification
								claimId={claimResponse.claim.claim_id}
								onSuccess={handleMFASuccess}
								onCancel={handleMFACancel}
							/>
						) : (
							<BiometricVerification
								claimId={claimResponse.claim.claim_id}
								onSuccess={handleMFASuccess}
								onCancel={handleMFACancel}
							/>
						)}
					</div>
				</Modal>
			)}
		</div>
	);
};

export default SubmitClaimPage;
