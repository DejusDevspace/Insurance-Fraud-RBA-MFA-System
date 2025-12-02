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
            const [statsData, riskData, alertsData, activityData] =
                await Promise.all([
                    adminService.getDashboardStats(),
                    adminService.getRiskDistribution(),
                    adminService.getFraudAlerts(10),
                    adminService.getRecentActivity(10),
                ]);

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
        <div className="min-h-screen bg-background">
            <Navbar />

            <div className="max-w-7xl mx-auto px-4 py-8">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-primary mb-2">
                        Admin Dashboard
                    </h1>
                    <p className="text-muted">
                        System overview and fraud detection analytics
                    </p>
                </div>

                {/* Key Metrics Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    {/* Total Claims */}
                    <Card hover>
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-muted text-sm mb-1">
                                    Total Claims
                                </p>
                                <p className="text-3xl font-bold text-primary">
                                    {stats.total_claims}
                                </p>
                                <div className="flex items-center gap-2 mt-2">
                                    <Badge variant="success">
                                        {stats.approved_claims} Approved
                                    </Badge>
                                </div>
                            </div>
                            <div className="p-3 rounded-lg bg-accent/10">
                                <FileText className="w-6 h-6 text-accent" />
                            </div>
                        </div>
                    </Card>

                    {/* Fraud Detection Rate */}
                    <Card hover>
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-muted text-sm mb-1">
                                    Fraud Rate
                                </p>
                                <p className="text-3xl font-bold text-error">
                                    {stats.fraud_rate.toFixed(1)}%
                                </p>
                                <div className="flex items-center gap-2 mt-2">
                                    <Badge variant="error">
                                        {stats.fraud_detected} Detected
                                    </Badge>
                                </div>
                            </div>
                            <div className="p-3 rounded-lg bg-error/10">
                                <AlertTriangle className="w-6 h-6 text-error" />
                            </div>
                        </div>
                    </Card>

                    {/* MFA Success Rate */}
                    <Card hover>
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-muted text-sm mb-1">
                                    MFA Success
                                </p>
                                <p className="text-3xl font-bold text-success">
                                    {stats.mfa_success_rate.toFixed(1)}%
                                </p>
                                <div className="flex items-center gap-2 mt-2">
                                    <Badge variant="info">
                                        {stats.mfa_triggered} Triggered
                                    </Badge>
                                </div>
                            </div>
                            <div className="p-3 rounded-lg bg-success/10">
                                <Shield className="w-6 h-6 text-success" />
                            </div>
                        </div>
                    </Card>

                    {/* Active Users */}
                    <Card hover>
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-muted text-sm mb-1">
                                    Active Users
                                </p>
                                <p className="text-3xl font-bold text-primary">
                                    {stats.active_users}
                                </p>
                                <div className="flex items-center gap-2 mt-2">
                                    <Badge variant="success">
                                        +{stats.new_users_today} Today
                                    </Badge>
                                </div>
                            </div>
                            <div className="p-3 rounded-lg bg-greenAccent/10">
                                <Users className="w-6 h-6 text-greenAccent" />
                            </div>
                        </div>
                    </Card>
                </div>

                {/* Claims Overview & Risk Distribution */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                    {/* Claims Status */}
                    <Card>
                        <div className="flex items-center gap-3 mb-6">
                            <div className="p-2 rounded-lg bg-accent/10">
                                <BarChart3 className="w-5 h-5 text-accent" />
                            </div>
                            <h2 className="text-xl font-semibold text-primary">
                                Claims Status
                            </h2>
                        </div>
                        <div className="space-y-4">
                            <div className="flex items-center justify-between p-3 rounded-lg bg-warning/10">
                                <div className="flex items-center gap-3">
                                    <Clock className="w-5 h-5 text-warning" />
                                    <span className="font-medium text-primary">
                                        Pending
                                    </span>
                                </div>
                                <span className="text-2xl font-bold text-warning">
                                    {stats.pending_claims}
                                </span>
                            </div>
                            <div className="flex items-center justify-between p-3 rounded-lg bg-success/10">
                                <div className="flex items-center gap-3">
                                    <CheckCircle className="w-5 h-5 text-success" />
                                    <span className="font-medium text-primary">
                                        Approved
                                    </span>
                                </div>
                                <span className="text-2xl font-bold text-success">
                                    {stats.approved_claims}
                                </span>
                            </div>
                            <div className="flex items-center justify-between p-3 rounded-lg bg-error/10">
                                <div className="flex items-center gap-3">
                                    <XCircle className="w-5 h-5 text-error" />
                                    <span className="font-medium text-primary">
                                        Rejected
                                    </span>
                                </div>
                                <span className="text-2xl font-bold text-error">
                                    {stats.rejected_claims}
                                </span>
                            </div>
                        </div>
                    </Card>

                    {/* Risk Distribution */}
                    <Card>
                        <div className="flex items-center gap-3 mb-6">
                            <div className="p-2 rounded-lg bg-accent/10">
                                <TrendingUp className="w-5 h-5 text-accent" />
                            </div>
                            <h2 className="text-xl font-semibold text-primary">
                                Risk Distribution
                            </h2>
                        </div>
                        <div className="space-y-4">
                            <div>
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm font-medium text-success">
                                        Low Risk
                                    </span>
                                    <span className="text-sm font-bold text-success">
                                        {riskDist.low_risk_percentage.toFixed(
                                            1
                                        )}
                                        %
                                    </span>
                                </div>
                                <div className="w-full h-2 bg-surface rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-success"
                                        style={{
                                            width: `${riskDist.low_risk_percentage}%`,
                                        }}
                                    />
                                </div>
                                <p className="text-xs text-muted mt-1">
                                    {riskDist.low_risk_count} claims
                                </p>
                            </div>
                            <div>
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm font-medium text-warning">
                                        Medium Risk
                                    </span>
                                    <span className="text-sm font-bold text-warning">
                                        {riskDist.medium_risk_percentage.toFixed(
                                            1
                                        )}
                                        %
                                    </span>
                                </div>
                                <div className="w-full h-2 bg-surface rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-warning"
                                        style={{
                                            width: `${riskDist.medium_risk_percentage}%`,
                                        }}
                                    />
                                </div>
                                <p className="text-xs text-muted mt-1">
                                    {riskDist.medium_risk_count} claims
                                </p>
                            </div>
                            <div>
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm font-medium text-error">
                                        High Risk
                                    </span>
                                    <span className="text-sm font-bold text-error">
                                        {riskDist.high_risk_percentage.toFixed(
                                            1
                                        )}
                                        %
                                    </span>
                                </div>
                                <div className="w-full h-2 bg-surface rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-error"
                                        style={{
                                            width: `${riskDist.high_risk_percentage}%`,
                                        }}
                                    />
                                </div>
                                <p className="text-xs text-muted mt-1">
                                    {riskDist.high_risk_count} claims
                                </p>
                            </div>
                            <div className="pt-3 border-t border-aux">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm text-muted">
                                        Average Risk Score
                                    </span>
                                    <span className="text-lg font-bold text-primary">
                                        {(
                                            riskDist.average_risk_score * 100
                                        ).toFixed(1)}
                                        %
                                    </span>
                                </div>
                            </div>
                        </div>
                    </Card>
                </div>

                {/* Fraud Alerts & Recent Activity */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                    {/* Fraud Alerts */}
                    <Card>
                        <div className="flex items-center gap-3 mb-6">
                            <div className="p-2 rounded-lg bg-error/10">
                                <AlertTriangle className="w-5 h-5 text-error" />
                            </div>
                            <h2 className="text-xl font-semibold text-primary">
                                Recent Fraud Alerts
                            </h2>
                        </div>
                        <div className="space-y-3">
                            {fraudAlerts.length === 0 ? (
                                <p className="text-sm text-muted text-center py-8">
                                    No fraud alerts at the moment
                                </p>
                            ) : (
                                fraudAlerts.slice(0, 5).map((alert) => (
                                    <div
                                        key={alert.detection_id}
                                        className="p-3 rounded-lg bg-error/5 border border-error/20 hover:border-error/40 transition-colors cursor-pointer"
                                    >
                                        <div className="flex items-start justify-between mb-2">
                                            <div>
                                                <p className="font-medium text-primary text-sm">
                                                    {alert.claim_number}
                                                </p>
                                                <p className="text-xs text-muted">
                                                    {alert.user_email}
                                                </p>
                                            </div>
                                            <Badge variant="error">
                                                {(
                                                    alert.fraud_probability *
                                                    100
                                                ).toFixed(0)}
                                                %
                                            </Badge>
                                        </div>
                                        <div className="flex items-center justify-between text-xs text-muted">
                                            <span className="capitalize">
                                                {alert.predicted_fraud_type?.replace(
                                                    /_/g,
                                                    " "
                                                )}
                                            </span>
                                            <span>
                                                {formatDate(alert.detected_at)}
                                            </span>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </Card>

                    {/* Recent Activity */}
                    <Card>
                        <div className="flex items-center gap-3 mb-6">
                            <div className="p-2 rounded-lg bg-accent/10">
                                <Activity className="w-5 h-5 text-accent" />
                            </div>
                            <h2 className="text-xl font-semibold text-primary">
                                Recent Activity
                            </h2>
                        </div>
                        <div className="space-y-3">
                            {recentActivity.length === 0 ? (
                                <p className="text-sm text-muted text-center py-8">
                                    No recent activity
                                </p>
                            ) : (
                                recentActivity.slice(0, 5).map((activity) => (
                                    <div
                                        key={activity.claim_id}
                                        className="p-3 rounded-lg bg-surface border border-aux hover:border-accent transition-colors cursor-pointer"
                                    >
                                        <div className="flex items-start justify-between mb-2">
                                            <div>
                                                <p className="font-medium text-primary text-sm">
                                                    {activity.claim_number}
                                                </p>
                                                <p className="text-xs text-muted">
                                                    {activity.user_email}
                                                </p>
                                            </div>
                                            {getRiskBadge(activity.risk_level)}
                                        </div>
                                        <div className="flex items-center justify-between text-xs">
                                            <span className="text-muted capitalize">
                                                {activity.claim_type.replace(
                                                    /_/g,
                                                    " "
                                                )}{" "}
                                                •{" "}
                                                {formatCurrency(
                                                    activity.claim_amount
                                                )}
                                            </span>
                                            <span className="text-muted">
                                                {formatDate(
                                                    activity.submitted_at
                                                )}
                                            </span>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </Card>
                </div>

                {/* Financial Overview */}
                <Card>
                    <div className="flex items-center gap-3 mb-6">
                        <div className="p-2 rounded-lg bg-greenAccent/10">
                            <BarChart3 className="w-5 h-5 text-greenAccent" />
                        </div>
                        <h2 className="text-xl font-semibold text-primary">
                            Financial Overview
                        </h2>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div>
                            <p className="text-muted text-sm mb-1">
                                Total Claim Amount
                            </p>
                            <p className="text-2xl font-bold text-primary">
                                {formatCurrency(stats.total_claim_amount)}
                            </p>
                        </div>
                        <div>
                            <p className="text-muted text-sm mb-1">
                                Average Claim Amount
                            </p>
                            <p className="text-2xl font-bold text-primary">
                                {formatCurrency(stats.avg_claim_amount)}
                            </p>
                        </div>
                        <div>
                            <p className="text-muted text-sm mb-1">
                                High Risk Claims
                            </p>
                            <p className="text-2xl font-bold text-error">
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
