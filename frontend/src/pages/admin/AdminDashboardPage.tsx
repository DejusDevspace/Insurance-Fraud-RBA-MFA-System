import React, { useEffect, useState } from "react";
// import { useNavigate } from "react-router-dom";
import { useNotification } from "../../hooks/useNotification";
import { adminService } from "../../services/adminService";
import { Card } from "../../components/common/Card";
import { Badge } from "../../components/common/Badge";
import { LoadingSpinner } from "../../components/common/LoadingSpinner";
import Navbar from "../../components/layout/Navbar";
import {
	BarChart3,
	FileText,
	AlertTriangle,
	Shield,
	Users,
	TrendingUp,
	CheckCircle,
	Clock,
	XCircle,
	Activity,
} from "lucide-react";
import type {
	DashboardStats,
	RiskDistribution,
	FraudAlert,
	RecentActivity,
} from "../../types/admin.types";

const AdminDashboardPage: React.FC = () => {
	// const navigate = useNavigate();
	const { showNotification } = useNotification();

	const [stats, setStats] = useState<DashboardStats | null>(null);
	const [riskDist, setRiskDist] = useState<RiskDistribution | null>(null);
	const [fraudAlerts, setFraudAlerts] = useState<FraudAlert[]>([]);
	const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([]);
	const [loading, setLoading] = useState(true);

	useEffect(() => {
		fetchDashboardData();
	}, []);

	const fetchDashboardData = async () => {
		try {
			const [statsData, riskData, alertsData, activityData] = await Promise.all(
				[
					adminService.getDashboardStats(),
					adminService.getRiskDistribution(),
					adminService.getFraudAlerts(10),
					adminService.getRecentActivity(10),
				],
			);

			setStats(statsData);
			setRiskDist(riskData);
			setFraudAlerts(alertsData);
			setRecentActivity(activityData);
		} catch (error: any) {
			showNotification("error", "Failed to load dashboard data");
		} finally {
			setLoading(false);
		}
	};

	const formatCurrency = (amount: number) => {
		return new Intl.NumberFormat("en-NG", {
			style: "currency",
			currency: "NGN",
			notation: "compact",
			maximumFractionDigits: 1,
		}).format(amount);
	};

	const formatDate = (dateString: string) => {
		return new Date(dateString).toLocaleDateString("en-US", {
			month: "short",
			day: "numeric",
			hour: "2-digit",
			minute: "2-digit",
		});
	};

	const getRiskBadge = (level: string) => {
		const variants: Record<string, "success" | "warning" | "error"> = {
			low: "success",
			medium: "warning",
			high: "error",
		};
		return <Badge variant={variants[level] || "info"}>{level}</Badge>;
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

	if (!stats || !riskDist) {
		return (
			<div className="min-h-screen bg-background">
				<Navbar />
				<div className="max-w-7xl mx-auto px-4 py-8">
					<Card>
						<div className="text-center py-12">
							<AlertTriangle className="w-16 h-16 text-error mx-auto mb-4" />
							<h2 className="text-xl font-semibold text-primary mb-2">
								Failed to Load Dashboard
							</h2>
							<p className="text-muted">
								Please refresh the page to try again.
							</p>
						</div>
					</Card>
				</div>
			</div>
		);
	}

	return (
		<div className="min-h-screen bg-background bg-mesh">
			<Navbar />

			<div className="max-w-7xl mx-auto px-4 pt-32 pb-12 relative z-10">
				{/* Header */}
				<div className="mb-12 animate-in fade-in slide-in-from-top-4 duration-700">
					<h1 className="text-4xl font-extrabold text-white mb-3">
						Admin <span className="my-gradient">Command Center</span>
					</h1>
					<p className="text-white/50 text-lg max-w-2xl">
						Real-time neural monitoring of insurance claims and fraud detection
						vectors.
					</p>
				</div>

				{/* Key Metrics Grid */}
				<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
					{/* Total Claims */}
					<Card variant="glass" hover className="border-accent/20">
						<div className="flex items-center justify-between">
							<div>
								<p className="text-white/40 text-[10px] font-black uppercase tracking-widest mb-2">
									Total Throughput
								</p>
								<p className="text-4xl font-black text-white">
									{stats.total_claims}
								</p>
								<div className="mt-3">
									<Badge variant="success">
										{stats.approved_claims} Verified
									</Badge>
								</div>
							</div>
							<div className="w-14 h-14 rounded-2xl bg-accent/20 flex items-center justify-center border border-accent/20 shadow-inner">
								<FileText className="w-7 h-7 text-accent" />
							</div>
						</div>
					</Card>

					{/* Fraud Detection Rate */}
					<Card variant="glass" hover className="border-error/20 bg-error/5!">
						<div className="flex items-center justify-between">
							<div>
								<p className="text-white/40 text-[10px] font-black uppercase tracking-widest mb-2">
									Fraud Vectors
								</p>
								<p className="text-4xl font-black text-error">
									{stats.fraud_rate.toFixed(1)}%
								</p>
								<div className="mt-3">
									<Badge variant="error" className="animate-pulse">
										{stats.fraud_detected} Positive
									</Badge>
								</div>
							</div>
							<div className="w-14 h-14 rounded-2xl bg-error/20 flex items-center justify-center border border-error/20">
								<AlertTriangle className="w-7 h-7 text-error" />
							</div>
						</div>
					</Card>

					{/* MFA Success Rate */}
					<Card
						variant="glass"
						hover
						className="border-success/20 bg-success/5!"
					>
						<div className="flex items-center justify-between">
							<div>
								<p className="text-white/40 text-[10px] font-black uppercase tracking-widest mb-2">
									Trust Score (MFA)
								</p>
								<p className="text-4xl font-black text-success">
									{stats.mfa_success_rate.toFixed(1)}%
								</p>
								<div className="mt-3">
									<Badge variant="info">{stats.mfa_triggered} Active</Badge>
								</div>
							</div>
							<div className="w-14 h-14 rounded-2xl bg-success/20 flex items-center justify-center border border-success/20">
								<Shield className="w-7 h-7 text-success" />
							</div>
						</div>
					</Card>

					{/* Active Users */}
					<Card variant="glass" hover className="border-white/10">
						<div className="flex items-center justify-between">
							<div>
								<p className="text-white/40 text-[10px] font-black uppercase tracking-widest mb-2">
									Neural Nodes
								</p>
								<p className="text-4xl font-black text-white">
									{stats.active_users}
								</p>
								<div className="mt-3 text-success font-black text-xs">
									+{stats.new_users_today} New Today
								</div>
							</div>
							<div className="w-14 h-14 rounded-2xl bg-white/5 flex items-center justify-center border border-white/10">
								<Users className="w-7 h-7 text-white/40" />
							</div>
						</div>
					</Card>
				</div>

				{/* Claims Overview & Risk Distribution */}
				<div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
					{/* Claims Status */}
					<Card variant="glass" className="border-white/10">
						<div className="flex items-center gap-4 mb-8">
							<div className="w-10 h-10 rounded-xl bg-accent/20 flex items-center justify-center border border-accent/20">
								<BarChart3 className="w-5 h-5 text-accent" />
							</div>
							<h2 className="text-2xl font-bold text-white">Status Overview</h2>
						</div>
						<div className="space-y-4">
							<div className="group flex items-center justify-between p-5 rounded-2xl bg-warning/5 border border-warning/10 hover:border-warning/30 transition-all">
								<div className="flex items-center gap-4">
									<Clock className="w-6 h-6 text-warning" />
									<span className="font-bold text-white text-lg">In Queue</span>
								</div>
								<span className="text-3xl font-black text-warning">
									{stats.pending_claims}
								</span>
							</div>
							<div className="group flex items-center justify-between p-5 rounded-2xl bg-success/5 border border-success/10 hover:border-success/30 transition-all">
								<div className="flex items-center gap-4">
									<CheckCircle className="w-6 h-6 text-success" />
									<span className="font-bold text-white text-lg">
										Finalized
									</span>
								</div>
								<span className="text-3xl font-black text-success">
									{stats.approved_claims}
								</span>
							</div>
							<div className="group flex items-center justify-between p-5 rounded-2xl bg-error/5 border border-error/10 hover:border-error/30 transition-all">
								<div className="flex items-center gap-4">
									<XCircle className="w-6 h-6 text-error" />
									<span className="font-bold text-white text-lg">Rejected</span>
								</div>
								<span className="text-3xl font-black text-error">
									{stats.rejected_claims}
								</span>
							</div>
						</div>
					</Card>

					{/* Risk Distribution */}
					<Card variant="glass" className="border-white/10">
						<div className="flex items-center gap-4 mb-8">
							<div className="w-10 h-10 rounded-xl bg-accent/20 flex items-center justify-center border border-accent/20">
								<TrendingUp className="w-5 h-5 text-accent" />
							</div>
							<h2 className="text-2xl font-bold text-white">Risk Topography</h2>
						</div>
						<div className="space-y-8">
							<div>
								<div className="flex items-center justify-between mb-3">
									<span className="text-sm font-bold text-success uppercase tracking-widest">
										Low Entropy
									</span>
									<span className="text-lg font-black text-success">
										{riskDist.low_risk_percentage.toFixed(1)}%
									</span>
								</div>
								<div className="w-full h-3 bg-white/5 rounded-full overflow-hidden border border-white/5">
									<div
										className="h-full bg-linear-to-r from-success/50 to-success shadow-[0_0_15px_rgba(74,222,128,0.5)] transition-all duration-1000"
										style={{ width: `${riskDist.low_risk_percentage}%` }}
									/>
								</div>
							</div>
							<div>
								<div className="flex items-center justify-between mb-3">
									<span className="text-sm font-bold text-warning uppercase tracking-widest">
										Moderate Risk
									</span>
									<span className="text-lg font-black text-warning">
										{riskDist.medium_risk_percentage.toFixed(1)}%
									</span>
								</div>
								<div className="w-full h-3 bg-white/5 rounded-full overflow-hidden border border-white/5">
									<div
										className="h-full bg-linear-to-r from-warning/50 to-warning shadow-[0_0_15px_rgba(245,158,11,0.5)] transition-all duration-1000"
										style={{ width: `${riskDist.medium_risk_percentage}%` }}
									/>
								</div>
							</div>
							<div>
								<div className="flex items-center justify-between mb-3">
									<span className="text-sm font-bold text-error uppercase tracking-widest">
										Critical Threat
									</span>
									<span className="text-lg font-black text-error">
										{riskDist.high_risk_percentage.toFixed(1)}%
									</span>
								</div>
								<div className="w-full h-3 bg-white/5 rounded-full overflow-hidden border border-white/5">
									<div
										className="h-full bg-linear-to-r from-error/50 to-error shadow-[0_0_15px_rgba(239,68,68,0.5)] transition-all duration-1000"
										style={{ width: `${riskDist.high_risk_percentage}%` }}
									/>
								</div>
							</div>
						</div>
					</Card>
				</div>

				{/* Fraud Alerts & Recent Activity */}
				<div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
					{/* Fraud Alerts */}
					<Card variant="glass" className="border-error/20">
						<div className="flex items-center gap-4 mb-8">
							<div className="w-10 h-10 rounded-xl bg-error/20 flex items-center justify-center border border-error/20">
								<AlertTriangle className="w-5 h-5 text-error" />
							</div>
							<h2 className="text-2xl font-bold text-white">
								Active Fraud Signals
							</h2>
						</div>
						<div className="space-y-4">
							{fraudAlerts.length === 0 ? (
								<div className="text-center py-20 text-white/20">
									<Shield className="w-12 h-12 mx-auto mb-4 opacity-10" />
									<p className="font-bold">System Secure. No Active Signals.</p>
								</div>
							) : (
								fraudAlerts.slice(0, 5).map((alert) => (
									<div
										key={alert.detection_id}
										className={`p-5 rounded-2xl bg-error/5 border transition-all cursor-pointer ${
											alert.fraud_probability > 0.8
												? "animate-pulse-red border-error/50 bg-error/10!"
												: "border-error/10 hover:border-error/30"
										}`}
									>
										<div className="flex items-start justify-between mb-4">
											<div>
												<p className="font-black text-white text-lg">
													{alert.claim_number}
												</p>
												<p className="text-xs font-medium text-white/40">
													ID: {alert.user_email}
												</p>
											</div>
											<div className="flex flex-col items-end">
												<span className="text-2xl font-black text-error">
													{(alert.fraud_probability * 100).toFixed(0)}%
												</span>
												<span className="text-[10px] font-black uppercase tracking-tighter text-error/60">
													Probability
												</span>
											</div>
										</div>
										<div className="flex items-center justify-between">
											<span className="px-3 py-1 rounded-full bg-error/20 text-error text-[10px] font-bold uppercase tracking-widest border border-error/20">
												{alert.predicted_fraud_type?.replace(/_/g, " ")}
											</span>
											<span className="text-[10px] font-medium text-white/40">
												{formatDate(alert.detected_at)}
											</span>
										</div>
									</div>
								))
							)}
						</div>
					</Card>

					{/* Recent Activity */}
					<Card variant="glass" className="border-white/10">
						<div className="flex items-center gap-4 mb-8">
							<div className="w-10 h-10 rounded-xl bg-accent/20 flex items-center justify-center border border-accent/20">
								<Activity className="w-5 h-5 text-accent" />
							</div>
							<h2 className="text-2xl font-bold text-white">
								Neural Output Stream
							</h2>
						</div>
						<div className="space-y-4">
							{recentActivity.length === 0 ? (
								<p className="text-sm text-white/20 text-center py-20 font-bold">
									No activity detected in stream.
								</p>
							) : (
								recentActivity.slice(0, 5).map((activity) => (
									<div
										key={activity.claim_id}
										className="group p-5 rounded-2xl bg-white/5 border border-white/5 hover:border-accent/30 hover:bg-white/10 transition-all cursor-pointer"
									>
										<div className="flex items-start justify-between mb-4">
											<div>
												<p className="font-bold text-white">
													{activity.claim_number}
												</p>
												<p className="text-xs font-medium text-white/40">
													{activity.user_email}
												</p>
											</div>
											{getRiskBadge(activity.risk_level)}
										</div>
										<div className="flex items-center justify-between text-[11px] font-bold">
											<div className="flex items-center gap-3">
												<span className="uppercase text-white/40 bg-white/5 px-2 py-0.5 rounded border border-white/5">
													{activity.claim_type.replace(/_/g, " ")}
												</span>
												<span className="text-accent">
													{formatCurrency(activity.claim_amount)}
												</span>
											</div>
											<span className="text-white/20">
												{formatDate(activity.submitted_at)}
											</span>
										</div>
									</div>
								))
							)}
						</div>
					</Card>
				</div>

				{/* Financial Overview */}
				<Card
					variant="glass"
					className="border-white/10 bg-linear-to-br from-white/5 to-transparent!"
				>
					<div className="flex items-center gap-4 mb-10">
						<div className="w-12 h-12 rounded-2xl bg-greenAccent/20 flex items-center justify-center border border-greenAccent/20 shadow-lg shadow-greenAccent/20">
							<BarChart3 className="w-6 h-6 text-greenAccent" />
						</div>
						<h2 className="text-2xl font-bold text-white">
							Neural Financial Analytics
						</h2>
					</div>
					<div className="grid grid-cols-1 md:grid-cols-3 gap-10">
						<div className="p-6 rounded-3xl bg-white/5 border border-white/5">
							<p className="text-white/40 text-[10px] font-black uppercase tracking-widest mb-3">
								Total Protocol Value
							</p>
							<p className="text-4xl font-black text-white">
								{formatCurrency(stats.total_claim_amount)}
							</p>
						</div>
						<div className="p-6 rounded-3xl bg-white/5 border border-white/5">
							<p className="text-white/40 text-[10px] font-black uppercase tracking-widest mb-3">
								Avg Claim Weight
							</p>
							<p className="text-4xl font-black text-white">
								{formatCurrency(stats.avg_claim_amount)}
							</p>
						</div>
						<div className="p-6 rounded-3xl bg-error/10 border border-error/10">
							<p className="text-error/60 text-[10px] font-black uppercase tracking-widest mb-3">
								Critical Exposure
							</p>
							<p className="text-4xl font-black text-error">
								{stats.high_risk_claims}
							</p>
						</div>
					</div>
				</Card>
			</div>
		</div>
	);
};

export default AdminDashboardPage;
