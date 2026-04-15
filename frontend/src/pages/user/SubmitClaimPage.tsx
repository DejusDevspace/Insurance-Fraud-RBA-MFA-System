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
		<div className="min-h-screen bg-background">
			<Navbar />

			<div className="max-w-4xl mx-auto px-4 py-8">
				{/* Header */}
				<div className="mb-8">
					<h1 className="text-3xl font-bold text-primary mb-2">
						Submit New Claim
					</h1>
					<p className="text-muted">
						Complete the form below to submit your insurance claim. Our
						intelligent system will assess the risk and may require additional
						verification.
					</p>
				</div>

				{/* Security Notice */}
				<Card className="mb-6 border-accent/50 bg-accent/5">
					<div className="flex items-start gap-4">
						<Shield className="w-6 h-6 text-accent shrink-0 mt-1" />
						<div>
							<h3 className="text-lg font-semibold text-primary mb-2">
								Risk-Based Authentication
							</h3>
							<p className="text-sm text-muted">
								Your claim will be automatically assessed for risk factors.
								High-risk claims may require additional verification (OTP or
								biometric) for enhanced security.
							</p>
						</div>
					</div>
				</Card>

				{/* Claim Form */}
				<Card>
					<form onSubmit={handleSubmit} className="space-y-6">
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
						>
							<option value="accident">Accident</option>
							<option value="theft">Theft</option>
							<option value="medical">Medical</option>
							<option value="property_damage">Property Damage</option>
							<option value="other">Other</option>
						</Select>

						{/* Claim Amount */}
						<Input
							label="Claim Amount"
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
									? formatCurrency(formData.claim_amount)
									: undefined
							}
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

						{/* Description */}
						<Textarea
							label="Claim Description"
							value={formData.claim_description}
							onChange={(e) =>
								handleChange("claim_description", e.target.value)
							}
							placeholder="Provide a detailed description of the incident..."
							disabled={loading}
							required
							rows={5}
							helperText="Minimum 10 characters required"
						/>

						{/* Supporting Documents Count */}
						<Input
							label="Number of Supporting Documents"
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
							helperText="Enter the number of documents you have prepared (0-10)"
						/>

						{/* Demo Controls Panel */}
						<div className="border border-aux rounded-lg overflow-hidden bg-background">
							<button
								type="button"
								onClick={() => setShowDemoControls(!showDemoControls)}
								className="w-full flex items-center justify-between p-4 bg-surface hover:bg-surface/80 transition-colors"
							>
								<div className="flex items-center gap-2 text-primary font-medium">
									<Settings className="w-5 h-5 text-accent" />
									Demo Controls (Context Overrides)
								</div>
								{showDemoControls ? (
									<ChevronUp className="w-5 h-5 text-muted" />
								) : (
									<ChevronDown className="w-5 h-5 text-muted" />
								)}
							</button>

							{showDemoControls && (
								<div className="p-4 space-y-4 border-t border-aux">
									<p className="text-sm text-muted">
										These options simulate context that the backend would normally extract automatically. Overriding these affects the generated Risk Assessment.
									</p>

									<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
										{/* Trusted Device */}
										<div className="flex items-center justify-between p-3 rounded bg-surface border border-aux">
											<label className="text-sm font-medium text-primary">Is Trusted Device?</label>
											<input
												type="checkbox"
												checked={formData.is_trusted_device || false}
												onChange={(e) => handleChange("is_trusted_device", e.target.checked)}
												className="w-4 h-4 text-accent border-aux focus:ring-accent rounded"
											/>
										</div>

										{/* Device Trust Score */}
										<div className="flex flex-col gap-1 p-3 rounded bg-surface border border-aux">
											<div className="flex justify-between items-center">
												<label className="text-sm font-medium text-primary">Device Trust Score</label>
												<span className="text-xs text-muted font-mono">{formData.device_trust_score ?? 0.5}</span>
											</div>
											<input
												type="range"
												min="0"
												max="1"
												step="0.1"
												value={formData.device_trust_score ?? 0.5}
												onChange={(e) => handleChange("device_trust_score", parseFloat(e.target.value))}
												className="w-full accent-accent"
											/>
										</div>

										{/* Geolocation Anomaly */}
										<div className="flex items-center justify-between p-3 rounded bg-surface border border-aux">
											<label className="text-sm font-medium text-primary">Is Geolocation Anomaly?</label>
											<input
												type="checkbox"
												checked={formData.is_geolocation_anomaly || false}
												onChange={(e) => handleChange("is_geolocation_anomaly", e.target.checked)}
												className="w-4 h-4 text-accent border-aux focus:ring-accent rounded"
											/>
										</div>

										{/* Geolocation Distance */}
										<div className="flex flex-col gap-1 p-3 rounded bg-surface border border-aux">
											<label className="text-sm font-medium text-primary">Geolocation Distance (km)</label>
											<input
												type="number"
												value={formData.geolocation_distance_km ?? 120}
												onChange={(e) => handleChange("geolocation_distance_km", parseFloat(e.target.value))}
												className="w-full px-3 py-1.5 text-sm bg-background border border-aux rounded text-primary focus:border-accent focus:ring-1 focus:ring-accent"
											/>
										</div>
									</div>
								</div>
							)}
						</div>

						{/* Info Notice */}
						<div className="flex items-start gap-3 p-4 rounded-lg bg-surface border border-aux">
							<AlertCircle className="w-5 h-5 text-accent shrink-0 mt-0.5" />
							<div className="text-sm text-muted">
								<p className="font-medium text-primary mb-1">
									Important Notice
								</p>
								<p>
									This is a demo system. Document uploads are simulated. In a
									production environment, you would upload actual supporting
									documents here.
								</p>
							</div>
						</div>

						{/* Submit Button */}
						<div className="flex gap-4 pt-4">
							<Button
								type="button"
								variant="secondary"
								onClick={() => navigate("/dashboard")}
								disabled={loading}
							>
								Cancel
							</Button>
							<Button
								type="submit"
								variant="primary"
								isLoading={loading}
								icon={<CheckCircle className="w-5 h-5" />}
								className="flex-1"
							>
								Submit Claim
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
					title="Additional Verification Required"
					size="md"
				>
					<div className="space-y-4">
						<div className="flex items-start gap-3 p-4 rounded-lg bg-warning/10 border border-warning">
							<AlertCircle className="w-5 h-5 text-warning shrink-0 mt-0.5" />
							<div className="text-sm">
								<p className="font-medium text-primary mb-1">
									Risk Assessment:{" "}
									{claimResponse.risk_assessment.risk_level.toUpperCase()}
								</p>
								<p className="text-muted">
									Your claim has been flagged for additional verification to
									ensure security. Please complete the verification below.
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
