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
	ChevronDown,
	ChevronUp,
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
	const [showDemoControls, setShowDemoControls] = useState(false);

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

			<div className="max-w-4xl mx-auto px-4 pt-32 pb-12 relative z-10">
				{/* Header */}
				<div className="mb-12 animate-in fade-in slide-in-from-top-4 duration-700">
					<h1 className="text-4xl font-extrabold text-white mb-3">
						Submit New <span className="my-gradient">Claim</span>
					</h1>
					<p className="text-white/50 text-lg">
						Complete the form below to submit your insurance claim. Our
						intelligent system will assess the risk and may require additional
						verification.
					</p>
				</div>

				{/* Security Notice */}
				<Card
					variant="glass"
					className="mb-12 border-accent/30 bg-accent/5! relative overflow-hidden group"
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
							<p className="text-white/60 text-lg max-w-2xl leading-relaxed">
								Your claim is being analyzed in real-time for behavioral and
								environmental risk factors. High-entropy claims may trigger
								additional{" "}
								<span className="text-accent font-bold">MFA protocols</span> for
								your protection.
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
								onChange={(e) => handleChange("incident_date", e.target.value)}
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
								handleChange("claim_amount", parseFloat(e.target.value) || 0)
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
							rows={6}
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

						{/* Demo Controls Panel */}
						<div className="rounded-2xl overflow-hidden bg-background/40 border border-white/5">
							<button
								type="button"
								onClick={() => setShowDemoControls(!showDemoControls)}
								className="w-full flex items-center justify-between p-5 hover:bg-white/5 transition-colors"
							>
								<div className="flex items-center gap-3 text-white/50 font-bold uppercase tracking-widest text-xs">
									<Settings className="w-4 h-4 text-accent" />
									Simulation Environment Overrides
								</div>
								{showDemoControls ? (
									<ChevronUp className="w-5 h-5 text-white/20" />
								) : (
									<ChevronDown className="w-5 h-5 text-white/20" />
								)}
							</button>

							{showDemoControls && (
								<div className="p-6 space-y-6 border-t border-white/5 bg-black/20">
									<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
										{/* Trusted Device */}
										<div className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/5">
											<label className="text-sm font-bold text-white/60">
												Device Integrity Certificate
											</label>
											<input
												type="checkbox"
												checked={formData.is_trusted_device || false}
												onChange={(e) =>
													handleChange("is_trusted_device", e.target.checked)
												}
												className="w-5 h-5 accent-accent"
											/>
										</div>

										{/* Device Trust Score */}
										<div className="flex flex-col gap-3 p-4 rounded-xl bg-white/5 border border-white/5">
											<div className="flex justify-between items-center">
												<label className="text-sm font-bold text-white/60">
													Trust Vector Magnitude
												</label>
												<span className="text-xs text-accent font-black font-mono">
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
												className="w-full accent-accent"
											/>
										</div>

										{/* Geolocation Anomaly */}
										<div className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/5">
											<label className="text-sm font-bold text-white/60">
												Geo-Spatial Anomaly
											</label>
											<input
												type="checkbox"
												checked={formData.is_geolocation_anomaly || false}
												onChange={(e) =>
													handleChange(
														"is_geolocation_anomaly",
														e.target.checked,
													)
												}
												className="w-5 h-5 accent-accent"
											/>
										</div>

										{/* Geolocation Distance */}
										<div className="flex flex-col gap-1 p-4 rounded-xl bg-white/5 border border-white/5">
											<label className="text-sm font-bold text-white/60">
												Anomalous Distance (km)
											</label>
											<input
												type="number"
												value={formData.geolocation_distance_km ?? 120}
												onChange={(e) =>
													handleChange(
														"geolocation_distance_km",
														parseFloat(e.target.value),
													)
												}
												className="w-full bg-transparent border-none text-accent font-black text-xl focus:ring-0 p-0"
											/>
										</div>
									</div>
								</div>
							)}
						</div>

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
