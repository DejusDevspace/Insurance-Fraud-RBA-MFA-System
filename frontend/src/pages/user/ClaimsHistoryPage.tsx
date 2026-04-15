import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useNotification } from "../../hooks/useNotification";
import { claimService } from "../../services/claimService";
import { Card } from "../../components/common/Card";
import { Badge } from "../../components/common/Badge";
import { Button } from "../../components/common/Button";
import { LoadingSpinner } from "../../components/common/LoadingSpinner";
import Navbar from "../../components/layout/Navbar";
import {
	FileText,
	Plus,
	AlertTriangle,
	ChevronRight,
	Filter,
	Shield,
} from "lucide-react";
import type { Claim } from "../../types/claim.types";

const ClaimsHistoryPage: React.FC = () => {
	const navigate = useNavigate();
	const { showNotification } = useNotification();

	const [claims, setClaims] = useState<Claim[]>([]);
	const [loading, setLoading] = useState(true);
	const [filterStatus, setFilterStatus] = useState<string>("all");

	useEffect(() => {
		fetchClaims();
	}, []);

	const fetchClaims = async () => {
		try {
			const data = await claimService.getUserClaims(50, 0);
			setClaims(data);
		} catch (error: any) {
			showNotification("error", "Failed to load claims");
		} finally {
			setLoading(false);
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
			month: "short",
			day: "numeric",
			hour: "2-digit",
			minute: "2-digit",
		});
	};

	const filteredClaims =
		filterStatus === "all"
			? claims
			: claims.filter((claim) => claim.claim_status === filterStatus);

	const statusCounts = claims.reduce(
		(acc, claim) => {
			acc[claim.claim_status] = (acc[claim.claim_status] || 0) + 1;
			return acc;
		},
		{} as Record<string, number>,
	);

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

	return (
		<div className="min-h-screen bg-background bg-mesh">
			<Navbar />

			<div className="max-w-7xl mx-auto px-4 pt-32 pb-12 relative z-10">
				{/* Header */}
				<div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-12 animate-in fade-in slide-in-from-top-4 duration-700">
					<div>
						<h1 className="text-4xl font-extrabold text-white mb-3">
							My <span className="my-gradient">Claims Repository</span>
						</h1>
						<p className="text-white/50 text-lg">
							Comprehensive archive of your insurance claim portfolio and status
							history.
						</p>
					</div>
					<Button
						variant="primary"
						onClick={() => navigate("/claims/submit")}
						className="h-12 px-6 font-bold shadow-lg shadow-accent/20"
					>
						<Plus className="w-5 h-5 mr-2" />
						Initiate New Claim
					</Button>
				</div>

				{/* MFA Pending Alert */}
				{claims.some((c) => c.requires_mfa && c.claim_status === "pending") && (
					<Card
						variant="glass"
						className="mb-12 border-warning/40 bg-warning/10! animate-pulse-red"
					>
						<div className="flex items-start gap-6">
							<div className="p-3 rounded-full bg-warning/20 border border-warning/30 shrink-0">
								<AlertTriangle className="w-8 h-8 text-warning" />
							</div>
							<div className="flex-1">
								<h3 className="text-xl font-bold text-white mb-2">
									Identity Challenges Detected
								</h3>
								<p className="text-white/60 mb-6 text-lg">
									{
										claims.filter(
											(c) => c.requires_mfa && c.claim_status === "pending",
										).length
									}{" "}
									claim(s) in your portfolio are currently under a security
									challenge. Cryptographic verification is required to authorize
									proceeding.
								</p>
							</div>
						</div>
					</Card>
				)}

				{/* Filters */}
				<div className="flex flex-col md:flex-row md:items-center gap-6 mb-12">
					<div className="flex items-center gap-2 text-white/40">
						<Filter className="w-5 h-5" />
						<span className="font-black uppercase tracking-widest text-[10px]">
							Filter Portfolio:
						</span>
					</div>
					<div className="flex flex-wrap gap-2">
						{[
							{
								value: "all",
								label: "All Records",
								count: claims.length,
							},
							{
								value: "pending",
								label: "Processing",
								count: statusCounts.pending || 0,
							},
							{
								value: "approved",
								label: "Authorized",
								count: statusCounts.approved || 0,
							},
							{
								value: "rejected",
								label: "Denied",
								count: statusCounts.rejected || 0,
							},
						].map((filter) => (
							<button
								key={filter.value}
								onClick={() => setFilterStatus(filter.value)}
								className={`px-6 py-2.5 rounded-xl text-sm font-bold transition-all duration-300 border ${
									filterStatus === filter.value
										? "bg-accent text-white border-accent shadow-lg shadow-accent/20"
										: "bg-white/5 text-white/40 border-white/5 hover:border-white/20 hover:text-white"
								}`}
							>
								{filter.label}{" "}
								<span className="opacity-40 ml-1">[{filter.count}]</span>
							</button>
						))}
					</div>
				</div>

				{/* Claims List */}
				{filteredClaims.length === 0 ? (
					<div className="py-24 text-center bg-white/5 rounded-3xl border border-dashed border-white/10">
						<div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mx-auto mb-6">
							<FileText className="w-10 h-10 text-white/20" />
						</div>
						<h3 className="text-2xl font-bold text-white mb-3">
							No Claims Found
						</h3>
						<p className="text-white/40 mb-10 max-w-md mx-auto">
							{filterStatus === "all"
								? "Your claim history is currently empty. Start your journey with InsureGuard AI today."
								: `There are no claims currently matching the '${filterStatus}' filtration criteria.`}
						</p>
						{filterStatus === "all" ? (
							<Button
								variant="primary"
								onClick={() => navigate("/claims/submit")}
								className="px-10 h-12 font-bold"
							>
								<Plus className="w-5 h-5 mr-2" />
								Start First Claim
							</Button>
						) : (
							<Button
								variant="secondary"
								onClick={() => setFilterStatus("all")}
								className="px-10 h-12 font-bold bg-white/5 border-white/10"
							>
								Reset Filters
							</Button>
						)}
					</div>
				) : (
					<div className="space-y-6">
						{filteredClaims.map((claim) => (
							<Card
								key={claim.claim_id}
								variant="glass"
								hover
								className="!p-0 cursor-pointer overflow-hidden group"
								onClick={() => navigate(`/claims/${claim.claim_id}`)}
							>
								<div className="flex items-stretch">
									{/* Status Bar */}
									<div
										className={`w-2 ${
											claim.claim_status === "approved"
												? "bg-success"
												: claim.claim_status === "pending"
													? "bg-warning"
													: claim.claim_status === "rejected"
														? "bg-error"
														: "bg-accent"
										}`}
									/>

									<div className="flex-1 p-6 md:p-8 flex items-center gap-8">
										{/* Icon */}
										<div className="hidden md:flex w-16 h-16 rounded-2xl bg-background/50 items-center justify-center border border-white/5 group-hover:bg-accent/10 transition-colors shrink-0">
											<FileText className="w-8 h-8 text-white/20 group-hover:text-accent transition-colors" />
										</div>

										{/* Content */}
										<div className="flex-1 min-w-0">
											<div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
												<div>
													<div className="flex items-center gap-3 mb-1">
														<h3 className="text-2xl font-black text-white tracking-tight">
															{claim.claim_number}
														</h3>
														{getStatusBadge(claim.claim_status)}
														{claim.requires_mfa &&
															claim.claim_status === "pending" && (
																<span className="flex items-center gap-1 text-[10px] font-black uppercase tracking-widest text-warning bg-warning/10 border border-warning/20 px-2 py-0.5 rounded">
																	<Shield className="w-3 h-3" />
																	Challenge
																</span>
															)}
													</div>
													<p className="text-white/40 font-bold uppercase tracking-widest text-[11px]">
														{claim.claim_type.replace(/_/g, " ")} • System Node:
														0x{claim.claim_id.slice(0, 4)}
													</p>
												</div>
												<div className="text-left md:text-right">
													<p className="text-3xl font-black text-white mb-1">
														{formatCurrency(claim.claim_amount)}
													</p>
													<p className="text-[11px] font-bold text-white/30 uppercase tracking-tighter">
														Authorized for processing on{" "}
														{formatDate(claim.submitted_at)}
													</p>
												</div>
											</div>

											{/* Description & Action */}
											<div className="flex items-end justify-between gap-6">
												<p className="text-white/50 text-base line-clamp-2 max-w-3xl leading-relaxed">
													{claim.claim_description}
												</p>
												<div className="hidden md:flex items-center gap-2 font-black text-[11px] uppercase tracking-widest text-accent group-hover:translate-x-2 transition-transform shrink-0">
													View Intelligence
													<ChevronRight className="w-4 h-4" />
												</div>
											</div>
										</div>
									</div>
								</div>
							</Card>
						))}
					</div>
				)}
			</div>
		</div>
	);
};

export default ClaimsHistoryPage;
