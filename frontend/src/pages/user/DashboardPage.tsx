import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../../hooks/useAuth";
import { useNotification } from "../../hooks/useNotification";
import { claimService } from "../../services/claimService";
import { Card } from "../../components/common/Card";
import { Button } from "../../components/common/Button";
import { Badge } from "../../components/common/Badge";
import { LoadingSpinner } from "../../components/common/LoadingSpinner";
import Navbar from "../../components/layout/Navbar";
import {
	FileText,
	Clock,
	CheckCircle,
	AlertTriangle,
	Plus,
	DollarSign,
	Shield,
} from "lucide-react";
import type { Claim } from "../../types/claim.types";

const DashboardPage: React.FC = () => {
	const navigate = useNavigate();
	const { user } = useAuth();
	const { showNotification } = useNotification();

	const [claims, setClaims] = useState<Claim[]>([]);
	const [loading, setLoading] = useState(true);
	const [stats, setStats] = useState({
		total: 0,
		pending: 0,
		approved: 0,
		rejected: 0,
		totalAmount: 0,
	});

	useEffect(() => {
		fetchClaims();
	}, []);

	const fetchClaims = async () => {
		try {
			const data = await claimService.getUserClaims(20, 0);
			setClaims(data);
			calculateStats(data);
		} catch (error: any) {
			showNotification("error", "Failed to load claims");
		} finally {
			setLoading(false);
		}
	};

	const calculateStats = (claimsData: Claim[]) => {
		const stats = claimsData.reduce(
			(acc, claim) => {
				acc.total += 1;
				acc.totalAmount += claim.claim_amount;

				switch (claim.claim_status) {
					case "pending":
						acc.pending += 1;
						break;
					case "approved":
						acc.approved += 1;
						break;
					case "rejected":
						acc.rejected += 1;
						break;
				}

				return acc;
			},
			{ total: 0, pending: 0, approved: 0, rejected: 0, totalAmount: 0 },
		);

		setStats(stats);
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
		});
	};

	if (loading) {
		return (
			<div className="min-h-screen bg-background">
				<Navbar />
				<div className="flex items-center justify-center h-[calc(100vh-4rem)]">
					<LoadingSpinner size="lg" />
				</div>
			</div>
		);
	}

	return (
		<div className="min-h-screen bg-background bg-mesh">
			<Navbar />

			<div className="max-w-7xl mx-auto px-4 pt-32 pb-12 relative z-10">
				{/* Welcome Section */}
				<div className="mb-12 animate-in fade-in slide-in-from-top-4 duration-700">
					<h1 className="text-4xl font-extrabold text-white mb-3">
						Welcome back,{" "}
						<span className="my-gradient">{user?.first_name}</span>!
					</h1>
					<p className="text-white/50 text-lg max-w-2xl">
						Monitor your insurance portfolio and track claim resolutions in
						real-time.
					</p>
				</div>

				{/* Quick Actions */}
				<div className="mb-12 flex gap-4">
					<Button
						variant="primary"
						onClick={() => navigate("/claims/submit")}
						className="h-14 px-8 text-lg font-bold group shadow-lg shadow-accent/20"
					>
						<Plus className="w-6 h-6 mr-2 group-hover:rotate-90 transition-transform" />
						Submit New Claim
					</Button>
				</div>

				{/* Stats Cards */}
				<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
					<Card variant="glass" hover className="border-accent/20 bg-accent/5!">
						<div className="flex items-center justify-between">
							<div>
								<p className="text-white/40 text-xs font-black uppercase tracking-widest mb-2">
									Total Claims
								</p>
								<p className="text-4xl font-black text-white">{stats.total}</p>
							</div>
							<div className="w-12 h-12 rounded-xl bg-accent/20 flex items-center justify-center border border-accent/30 shadow-inner">
								<FileText className="w-6 h-6 text-accent" />
							</div>
						</div>
					</Card>

					<Card
						variant="glass"
						hover
						className="border-warning/20 bg-warning/5!"
					>
						<div className="flex items-center justify-between">
							<div>
								<p className="text-white/40 text-xs font-black uppercase tracking-widest mb-2">
									In Processing
								</p>
								<p className="text-4xl font-black text-warning">
									{stats.pending}
								</p>
							</div>
							<div className="w-12 h-12 rounded-xl bg-warning/20 flex items-center justify-center border border-warning/30">
								<Clock className="w-6 h-6 text-warning" />
							</div>
						</div>
					</Card>

					<Card
						variant="glass"
						hover
						className="border-success/20 bg-success/5!"
					>
						<div className="flex items-center justify-between">
							<div>
								<p className="text-white/40 text-xs font-black uppercase tracking-widest mb-2">
									Total Approved
								</p>
								<p className="text-4xl font-black text-success">
									{stats.approved}
								</p>
							</div>
							<div className="w-12 h-12 rounded-xl bg-success/20 flex items-center justify-center border border-success/30">
								<CheckCircle className="w-6 h-6 text-success" />
							</div>
						</div>
					</Card>

					<Card variant="glass" hover className="border-white/10">
						<div className="flex items-center justify-between">
							<div>
								<p className="text-white/40 text-xs font-black uppercase tracking-widest mb-2">
									Estimated Value
								</p>
								<p className="text-3xl font-black text-white">
									{formatCurrency(stats.totalAmount)}
								</p>
							</div>
							<div className="w-12 h-12 rounded-xl bg-white/5 flex items-center justify-center border border-white/10">
								<DollarSign className="w-6 h-6 text-white/60" />
							</div>
						</div>
					</Card>
				</div>

				{/* MFA Alerts */}
				{claims.some((c) => c.requires_mfa && c.claim_status === "pending") && (
					<Card
						variant="glass"
						className="mb-12 border-warning/40 bg-warning/10! animate-pulse-red relative overflow-hidden"
					>
						<div className="absolute top-0 right-0 p-4">
							<AlertTriangle className="w-12 h-12 text-warning/20 -mr-4 -mt-4 rotate-12" />
						</div>
						<div className="flex items-start gap-6">
							<div className="p-3 rounded-full bg-warning/20 border border-warning/30 shrink-0">
								<Shield className="w-8 h-8 text-warning" />
							</div>
							<div className="flex-1">
								<h3 className="text-xl font-bold text-white mb-2">
									Identity Verification Required
								</h3>
								<p className="text-white/60 mb-6 max-w-2xl text-lg">
									We've detected that{" "}
									<span className="text-warning font-bold">
										{
											claims.filter(
												(c) => c.requires_mfa && c.claim_status === "pending",
											).length
										}{" "}
										claim(s)
									</span>{" "}
									require Multi-Factor Authentication to proceed with approval.
								</p>
								<Button
									variant="danger"
									onClick={() => navigate("/claims/history")}
									className="px-6 py-2 rounded-lg font-bold"
								>
									Verify Identity Now
								</Button>
							</div>
						</div>
					</Card>
				)}

				{/* Recent Claims Section */}
				<Card variant="glass" className="border-white/10">
					<div className="flex items-center justify-between mb-8">
						<div>
							<h2 className="text-2xl font-bold text-white">Recent Activity</h2>
							<p className="text-white/40 text-sm">
								Your most recently submitted claims and their status.
							</p>
						</div>
						<Button
							variant="secondary"
							onClick={() => navigate("/claims/history")}
							className="bg-white/5 hover:bg-white/10"
						>
							History
						</Button>
					</div>

					{claims.length === 0 ? (
						<div className="text-center py-20 bg-white/5 rounded-2xl border border-dashed border-white/10">
							<div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mx-auto mb-6">
								<FileText className="w-10 h-10 text-white/20" />
							</div>
							<h3 className="text-xl font-bold text-white mb-2">Clean Slate</h3>
							<p className="text-white/40 mb-8">
								Ready to submit your first insurance claim?
							</p>
							<Button
								variant="primary"
								onClick={() => navigate("/claims/submit")}
								icon={<Plus className="w-5 h-5" />}
							>
								Get Started
							</Button>
						</div>
					) : (
						<div className="space-y-4">
							{claims.slice(0, 5).map((claim) => (
								<div
									key={claim.claim_id}
									className="group flex items-center justify-between p-5 rounded-2xl border border-white/5 bg-white/5 hover:bg-white/10 hover:border-accent/30 transition-all duration-300 cursor-pointer"
									onClick={() => navigate(`/claims/${claim.claim_id}`)}
								>
									<div className="flex items-center gap-5 flex-1">
										<div className="w-14 h-14 rounded-2xl bg-background/50 flex items-center justify-center border border-white/5 group-hover:bg-accent/10 transition-colors">
											<FileText className="w-7 h-7 text-white/40 group-hover:text-accent transition-colors" />
										</div>
										<div className="flex-1">
											<div className="flex items-center gap-3 mb-1">
												<p className="font-bold text-white text-lg">
													{claim.claim_number}
												</p>
												{getStatusBadge(claim.claim_status)}
											</div>
											<div className="flex items-center gap-3 text-sm font-medium text-white/40">
												<span className="uppercase tracking-widest text-[10px] bg-white/5 px-2 py-0.5 rounded border border-white/5">
													{claim.claim_type}
												</span>
												<span>•</span>
												<span>{formatDate(claim.submitted_at)}</span>
											</div>
										</div>
									</div>
									<div className="text-right">
										<p className="text-2xl font-black text-white">
											{formatCurrency(claim.claim_amount)}
										</p>
										{claim.requires_mfa && claim.claim_status === "pending" && (
											<span className="inline-flex items-center gap-1.5 text-[10px] font-black uppercase tracking-tighter text-warning bg-warning/10 px-2 py-0.5 rounded-full border border-warning/20 mt-1">
												<Shield className="w-3 h-3" />
												MFA Required
											</span>
										)}
									</div>
								</div>
							))}
						</div>
					)}
				</Card>
			</div>
		</div>
	);
};

export default DashboardPage;
