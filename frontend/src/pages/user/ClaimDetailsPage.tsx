import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useNotification } from "../../hooks/useNotification";
import { claimService } from "../../services/claimService";
import { riskService } from "../../services/riskService";
import { fraudService } from "../../services/fraudService";
import { Card } from "../../components/common/Card";
import { Badge } from "../../components/common/Badge";
import { Button } from "../../components/common/Button";
import { LoadingSpinner } from "../../components/common/LoadingSpinner";
import { Modal } from "../../components/common/Modal";
import Navbar from "../../components/layout/Navbar";
import OTPVerification from "../../components/mfa/OTPVerification";
import BiometricVerification from "../../components/mfa/BiometricVerification";
import RiskScoreDisplay from "../../components/risk/RiskScoreDisplay";
import FraudAlert from "../../components/fraud/FraudAlert";
import {
	FileText,
	// Calendar,
	// DollarSign,
	AlertTriangle,
	CheckCircle,
	XCircle,
	ArrowLeft,
	Shield,
	// Info,
} from "lucide-react";
import type { Claim } from "../../types/claim.types";
import type { RiskScore } from "../../types/risk.types";
import type { FraudDetection } from "../../types/fraud.types";

const ClaimDetailsPage: React.FC = () => {
	const { claimId } = useParams<{ claimId: string }>();
	const navigate = useNavigate();
	const { showNotification } = useNotification();

	const [claim, setClaim] = useState<Claim | null>(null);
	const [riskScore, setRiskScore] = useState<RiskScore | null>(null);
	const [fraudDetection, setFraudDetection] = useState<FraudDetection | null>(
		null,
	);
	const [loading, setLoading] = useState(true);
	const [showMFAModal, setShowMFAModal] = useState(false);

	useEffect(() => {
		if (claimId) {
			fetchClaimDetails();
		}
	}, [claimId]);

	const fetchClaimDetails = async () => {
		try {
			const [claimData, riskData, fraudData] = await Promise.all([
				claimService.getClaimById(claimId!),
				riskService.getRiskScore(claimId!),
				fraudService.getFraudDetection(claimId!),
			]);

			setClaim(claimData);
			setRiskScore(riskData);
			setFraudDetection(fraudData);
		} catch (error: any) {
			console.error("Error fetching claim details:", error);

			// Try to fetch claim and risk data without fraud
			try {
				const [claimData, riskData] = await Promise.all([
					claimService.getClaimById(claimId!),
					riskService.getRiskScore(claimId!).catch(() => null),
				]);

				setClaim(claimData);
				setRiskScore(riskData);

				// Log fraud API failure for debugging
				console.warn(
					"Fraud detection data unavailable:",
					error.response?.data?.detail || error.message,
				);
			} catch (claimError: any) {
				showNotification("error", "Failed to load claim details");
				navigate("/claims/history");
			}
		} finally {
			setLoading(false);
		}
	};

	const handleMFASuccess = () => {
		setShowMFAModal(false);
		showNotification(
			"success",
			"Verification successful! Claim has been processed.",
		);
		fetchClaimDetails(); // Refresh claim data
	};

	const getStatusIcon = (status: string) => {
		switch (status) {
			case "approved":
				return <CheckCircle className="w-6 h-6 text-success" />;
			case "rejected":
				return <XCircle className="w-6 h-6 text-error" />;
			case "pending":
				return <AlertTriangle className="w-6 h-6 text-warning" />;
			default:
				return <FileText className="w-6 h-6 text-accent" />;
		}
	};

	const getStatusBadge = (status: string) => {
		const variants: Record<string, "success" | "warning" | "error" | "info"> = {
			approved: "success",
			pending: "warning",
			rejected: "error",
			under_review: "info",
		};

		return <Badge variant={variants[status] || "info"}>{status}</Badge>;
	};

	const formatCurrency = (amount: number) => {
		return new Intl.NumberFormat("en-NG", {
			style: "currency",
			currency: "NGN",
		}).format(amount);
	};

	const formatDate = (dateString: string) => {
		return new Date(dateString).toLocaleDateString("en-US", {
			year: "numeric",
			month: "long",
			day: "numeric",
			hour: "2-digit",
			minute: "2-digit",
		});
	};

	if (loading) {
		return (
			<div className="min-h-screen bg-background bg-mesh flex flex-col">
				<Navbar />
				<div className="flex-1 flex items-center justify-center">
					<LoadingSpinner size="lg" />
				</div>
			</div>
		);
	}

	if (!claim) {
		return (
			<div className="min-h-screen bg-background bg-mesh">
				<Navbar />
				<div className="max-w-7xl mx-auto px-4 pt-40 pb-12 relative z-10 text-center">
					<Card
						variant="glass"
						className="max-w-md mx-auto py-16 border-error/20"
					>
						<AlertTriangle className="w-20 h-20 text-error/40 mx-auto mb-6" />
						<h2 className="text-3xl font-black text-white mb-4 tracking-tight">
							Record Not Found
						</h2>
						<p className="text-white/40 mb-10 text-lg">
							The neural signature for this claim ID could not be located in the
							Insurance Fraud repository.
						</p>
						<Button
							variant="primary"
							onClick={() => navigate("/claims/history")}
							className="h-12 px-8 font-bold"
						>
							Back to Repository
						</Button>
					</Card>
				</div>
			</div>
		);
	}

	return (
		<div className="min-h-screen bg-background bg-mesh">
			<Navbar />

			<div className="max-w-7xl mx-auto px-4 pt-32 pb-12 relative z-10">
				{/* Back Button */}
				<button
					onClick={() => navigate("/claims/history")}
					className="flex items-center gap-2 text-white/40 hover:text-white transition-colors mb-8 group font-bold uppercase tracking-widest text-[10px]"
				>
					<ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
					Back to History
				</button>

				{/* Claim Header */}
				<Card
					variant="glass"
					className="mb-8 border-white/10 relative overflow-hidden"
				>
					<div className="flex flex-col md:flex-row md:items-center justify-between gap-8">
						<div className="flex items-start gap-6">
							<div className="w-16 h-16 rounded-2xl bg-background/50 flex items-center justify-center border border-white/5 ring-4 ring-white/5">
								{getStatusIcon(claim.claim_status)}
							</div>
							<div>
								<h1 className="text-4xl font-black text-white tracking-tight mb-2">
									{claim.claim_number}
								</h1>
								<div className="flex items-center gap-3">
									{getStatusBadge(claim.claim_status)}
									{claim.requires_mfa && claim.claim_status === "pending" && (
										<span className="flex items-center gap-1.5 text-[10px] font-black uppercase tracking-widest text-warning bg-warning/10 border border-warning/20 px-2 py-0.5 rounded">
											<Shield className="w-3 h-3" />
											MFA Required
										</span>
									)}
								</div>
							</div>
						</div>
						<div className="flex flex-col md:items-end">
							<p className="text-3xl font-black text-white mb-1">
								{formatCurrency(claim.claim_amount)}
							</p>
							<p className="text-[11px] font-bold text-white/30 uppercase tracking-widest">
								Transaction ID: 0x{claim.claim_id.slice(0, 12)}
							</p>
						</div>
					</div>

					{/* MFA Action */}
					{claim.requires_mfa && claim.claim_status === "pending" && (
						<div className="mt-10 p-6 rounded-2xl bg-warning/10 border border-warning/20 animate-pulse-red relative overflow-hidden">
							<div className="absolute top-0 right-0 p-4 opacity-5">
								<Shield className="w-24 h-24 text-warning -mr-8 -mt-8" />
							</div>
							<div className="flex items-start gap-6 relative z-10">
								<div className="w-12 h-12 rounded-full bg-warning/20 border border-warning/30 flex items-center justify-center shrink-0">
									<Shield className="w-6 h-6 text-warning" />
								</div>
								<div className="flex-1">
									<h3 className="text-xl font-bold text-white mb-2">
										Action Required: Multi-Factor Authentication
									</h3>
									<p className="text-white/60 mb-6 text-base max-w-2xl leading-relaxed">
										This high-entropy claim requires a biometric or
										cryptographic verification challenge to proceed with final
										authorization.
									</p>
									<Button
										variant="danger"
										onClick={() => setShowMFAModal(true)}
										className="h-10 px-6 font-bold shadow-lg shadow-error/20"
									>
										Initiate Verification
									</Button>
								</div>
							</div>
						</div>
					)}

					{/* Claim Details Grid */}
					<div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-10 mt-12 pb-10 border-b border-white/5">
						<div>
							<p className="text-white/30 text-[10px] font-black uppercase tracking-widest mb-3 italic">
								Claim Classification
							</p>
							<p className="text-lg font-bold text-white capitalize">
								{claim.claim_type.replace(/_/g, " ")}
							</p>
						</div>
						<div>
							<p className="text-white/30 text-[10px] font-black uppercase tracking-widest mb-3 italic">
								Claim Amount
							</p>
							<p className="text-lg font-bold text-white">
								{formatCurrency(claim.claim_amount)}
							</p>
						</div>
						<div>
							<p className="text-white/30 text-[10px] font-black uppercase tracking-widest mb-3 italic">
								Incident Date
							</p>
							<p className="text-lg font-bold text-white">
								{new Date(claim.incident_date).toLocaleDateString("en-US", {
									year: "numeric",
									month: "short",
									day: "numeric",
								})}
							</p>
						</div>
						<div>
							<p className="text-white/30 text-[10px] font-black uppercase tracking-widest mb-3 italic">
								System Ingestion
							</p>
							<p className="text-md font-bold text-white">
								{formatDate(claim.submitted_at)}
							</p>
						</div>
					</div>

					{/* Description */}
					<div className="mt-10">
						<h3 className="text-sm font-black text-white/40 uppercase tracking-[0.2em] mb-4">
							Claim Description
						</h3>
						<div className="bg-background/40 p-6 rounded-2xl border border-white/5 text-white/70 leading-loose text-lg font-medium italic">
							"{claim.claim_description}"
						</div>
					</div>

					{/* Supporting Documents */}
					<div className="mt-8 flex items-center justify-between p-4 bg-white/5 rounded-xl border border-white/5">
						<div className="flex items-center gap-3">
							<FileText className="w-5 h-5 text-accent" />
							<span className="text-sm font-bold text-white/60 uppercase tracking-widest">
								Supporting Documents
							</span>
						</div>
						<span className="text-lg font-black text-white">
							{claim.supporting_documents_count} Files Linked
						</span>
					</div>

					{/* Rejection Reason */}
					{claim.claim_status === "rejected" && claim.rejection_reason && (
						<div className="mt-10 p-6 rounded-2xl bg-error/10 border border-error/20 animate-in shake">
							<div className="flex items-start gap-4">
								<div className="w-10 h-10 rounded-full bg-error/20 flex items-center justify-center shrink-0">
									<XCircle className="w-6 h-6 text-error" />
								</div>
								<div>
									<h3 className="text-xl font-bold text-white mb-2">
										Rejection Reason
									</h3>
									<p className="text-white/60 text-base leading-relaxed">
										{claim.rejection_reason}
									</p>
								</div>
							</div>
						</div>
					)}
				</Card>

				{/* AI Analysis Section */}
				<div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
					{/* Risk Assessment */}
					{riskScore && (
						<div className="animate-in fade-in slide-in-from-left-4 duration-700">
							<RiskScoreDisplay
								claimId={claim.claim_id}
								riskScore={riskScore}
							/>
						</div>
					)}

					{/* Fraud Detection */}
					{fraudDetection && (
						<div className="animate-in fade-in slide-in-from-right-4 duration-700">
							<FraudAlert
								claimId={claim.claim_id}
								fraudDetection={fraudDetection}
							/>
						</div>
					)}
				</div>

				{/* Processing Timeline */}
				<Card variant="glass" className="border-white/10">
					<h2 className="text-2xl font-black text-white mb-10 uppercase tracking-widest">
						Claim Processing Timeline
					</h2>
					<div className="relative space-y-12 before:absolute before:left-[11px] before:top-2 before:bottom-2 before:w-0.5px before:bg-white/5">
						<div className="flex items-start gap-8 relative z-10 transition-all hover:translate-x-1">
							<div className="w-6 h-6 rounded-full bg-accent ring-8 ring-accent/10 mt-1.5 flex items-center justify-center border-4 border-background"></div>
							<div className="flex-1">
								<p className="text-xl font-black text-white mb-1">
									Claim Submitted
								</p>
								<p className="text-sm font-bold text-white/30 uppercase tracking-widest">
									{formatDate(claim.submitted_at)}
								</p>
							</div>
						</div>
						{claim.processed_at && (
							<div className="flex items-start gap-8 relative z-10 transition-all hover:translate-x-1">
								<div className="w-6 h-6 rounded-full bg-success ring-8 ring-success/10 mt-1.5 flex items-center justify-center border-4 border-background"></div>
								<div className="flex-1">
									<p className="text-xl font-black text-white mb-1">
										Claim Processed
									</p>
									<p className="text-sm font-bold text-white/30 uppercase tracking-widest">
										{formatDate(claim.processed_at)}
									</p>
								</div>
							</div>
						)}
					</div>
				</Card>
			</div>

			{/* MFA Modal */}
			{showMFAModal && (
				<Modal
					isOpen={showMFAModal}
					onClose={() => setShowMFAModal(false)}
					title="Neural Identity Challenge"
					size="md"
				>
					<div className="p-2">
						{claim.mfa_method === "otp" ? (
							<OTPVerification
								claimId={claim.claim_id}
								onSuccess={handleMFASuccess}
								onCancel={() => setShowMFAModal(false)}
							/>
						) : (
							<BiometricVerification
								claimId={claim.claim_id}
								onSuccess={handleMFASuccess}
								onCancel={() => setShowMFAModal(false)}
							/>
						)}
					</div>
				</Modal>
			)}
		</div>
	);
};

export default ClaimDetailsPage;
