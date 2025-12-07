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
            { total: 0, pending: 0, approved: 0, rejected: 0, totalAmount: 0 }
        );

        setStats(stats);
    };

    const getStatusBadge = (status: string) => {
        const variants: Record<
            string,
            "success" | "warning" | "error" | "info"
        > = {
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
        <div className="min-h-screen bg-background">
            <Navbar />

            <div className="max-w-7xl mx-auto px-4 py-8">
                {/* Welcome Section */}
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-primary mb-2">
                        Welcome back, {user?.first_name}!
                    </h1>
                    <p className="text-muted">
                        Manage your insurance claims and view your account
                        status
                    </p>
                </div>

                {/* Stats Cards */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                    <Card hover>
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-muted text-sm mb-1">
                                    Total Claims
                                </p>
                                <p className="text-3xl font-bold text-primary">
                                    {stats.total}
                                </p>
                            </div>
                            <div className="p-3 rounded-lg bg-accent/10">
                                <FileText className="w-6 h-6 text-accent" />
                            </div>
                        </div>
                    </Card>

                    <Card hover>
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-muted text-sm mb-1">
                                    Pending
                                </p>
                                <p className="text-3xl font-bold text-warning">
                                    {stats.pending}
                                </p>
                            </div>
                            <div className="p-3 rounded-lg bg-warning/10">
                                <Clock className="w-6 h-6 text-warning" />
                            </div>
                        </div>
                    </Card>

                    <Card hover>
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-muted text-sm mb-1">
                                    Approved
                                </p>
                                <p className="text-3xl font-bold text-success">
                                    {stats.approved}
                                </p>
                            </div>
                            <div className="p-3 rounded-lg bg-success/10">
                                <CheckCircle className="w-6 h-6 text-success" />
                            </div>
                        </div>
                    </Card>

                    <Card hover>
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-muted text-sm mb-1">
                                    Total Amount
                                </p>
                                <p className="text-2xl font-bold text-primary">
                                    {formatCurrency(stats.totalAmount)}
                                </p>
                            </div>
                            <div className="p-3 rounded-lg bg-greenAccent/10">
                                <DollarSign className="w-6 h-6 text-greenAccent" />
                            </div>
                        </div>
                    </Card>
                </div>

                {/* Quick Actions */}
                <div className="mb-8">
                    <Button
                        variant="primary"
                        size="lg"
                        onClick={() => navigate("/claims/submit")}
                        icon={<Plus className="w-5 h-5" />}
                    >
                        Submit New Claim
                    </Button>
                </div>

                {/* MFA Alerts */}
                {claims.some(
                    (c) => c.requires_mfa && c.claim_status === "pending"
                ) && (
                    <Card className="mb-8 border-warning bg-warning/5">
                        <div className="flex items-start gap-4">
                            <AlertTriangle className="w-6 h-6 text-warning shrink-0 mt-1" />
                            <div className="flex-1">
                                <h3 className="text-lg font-semibold text-primary mb-2">
                                    Action Required: MFA Verification
                                </h3>
                                <p className="text-muted mb-4">
                                    You have{" "}
                                    {
                                        claims.filter(
                                            (c) =>
                                                c.requires_mfa &&
                                                c.claim_status === "pending"
                                        ).length
                                    }{" "}
                                    claim(s) requiring multi-factor
                                    authentication to complete processing.
                                </p>
                                <Button
                                    variant="danger"
                                    size="sm"
                                    onClick={() => navigate("/claims/history")}
                                >
                                    View Pending Claims
                                </Button>
                            </div>
                        </div>
                    </Card>
                )}

                {/* Recent Claims */}
                <Card>
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-xl font-semibold text-primary">
                            Recent Claims
                        </h2>
                        <Button
                            variant="secondary"
                            size="sm"
                            onClick={() => navigate("/claims/history")}
                        >
                            View All
                        </Button>
                    </div>

                    {claims.length === 0 ? (
                        <div className="text-center py-12">
                            <FileText className="w-16 h-16 text-muted mx-auto mb-4 opacity-50" />
                            <p className="text-muted mb-4">No claims yet</p>
                            <Button
                                variant="primary"
                                onClick={() => navigate("/claims/submit")}
                                icon={<Plus className="w-5 h-5" />}
                            >
                                Submit Your First Claim
                            </Button>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            {claims.slice(0, 5).map((claim) => (
                                <div
                                    key={claim.claim_id}
                                    className="flex items-center justify-between p-4 rounded-lg border border-aux hover:border-accent transition-colors cursor-pointer"
                                    onClick={() =>
                                        navigate(`/claims/${claim.claim_id}`)
                                    }
                                >
                                    <div className="flex items-center gap-4 flex-1">
                                        <div className="p-3 rounded-lg bg-surface">
                                            <FileText className="w-5 h-5 text-accent" />
                                        </div>
                                        <div className="flex-1">
                                            <div className="flex items-center gap-2 mb-1">
                                                <p className="font-medium text-primary">
                                                    {claim.claim_number}
                                                </p>
                                                {getStatusBadge(
                                                    claim.claim_status
                                                )}
                                            </div>
                                            <p className="text-sm text-muted">
                                                {claim.claim_type} •{" "}
                                                {formatDate(claim.submitted_at)}
                                            </p>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <p className="font-semibold text-primary">
                                            {formatCurrency(claim.claim_amount)}
                                        </p>
                                        {claim.requires_mfa &&
                                            claim.claim_status ===
                                                "pending" && (
                                                <p className="text-xs text-warning mt-1">
                                                    MFA Required
                                                </p>
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
