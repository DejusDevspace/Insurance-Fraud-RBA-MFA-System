import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useNotification } from "../../hooks/useNotification";
import { claimService } from "../../services/claimService";
import { Card } from "../../components/common/Card";
import { Badge } from "../../components/common/Badge";
import { Button } from "../../components/common/Button";
import { LoadingSpinner } from "../../components/common/LoadingSpinner";
import { EmptyState } from "../../components/common/EmptyState";
import Navbar from "../../components/layout/Navbar";
import {
    FileText,
    Plus,
    Calendar,
    DollarSign,
    AlertTriangle,
    ChevronRight,
    Filter,
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
            hour: "2-digit",
            minute: "2-digit",
        });
    };

    const filteredClaims =
        filterStatus === "all"
            ? claims
            : claims.filter((claim) => claim.claim_status === filterStatus);

    const statusCounts = claims.reduce((acc, claim) => {
        acc[claim.claim_status] = (acc[claim.claim_status] || 0) + 1;
        return acc;
    }, {} as Record<string, number>);

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
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <h1 className="text-3xl font-bold text-primary mb-2">
                            My Claims
                        </h1>
                        <p className="text-muted">
                            View and manage all your insurance claims
                        </p>
                    </div>
                    <Button
                        variant="primary"
                        onClick={() => navigate("/claims/submit")}
                        icon={<Plus className="w-5 h-5" />}
                    >
                        Submit New Claim
                    </Button>
                </div>

                {/* MFA Pending Alert */}
                {claims.some(
                    (c) => c.requires_mfa && c.claim_status === "pending"
                ) && (
                    <Card className="mb-6 border-warning bg-warning/5">
                        <div className="flex items-start gap-4">
                            <AlertTriangle className="w-6 h-6 text-warning shrink-0 mt-1" />
                            <div className="flex-1">
                                <h3 className="text-lg font-semibold text-primary mb-2">
                                    Pending MFA Verification
                                </h3>
                                <p className="text-muted mb-4">
                                    {
                                        claims.filter(
                                            (c) =>
                                                c.requires_mfa &&
                                                c.claim_status === "pending"
                                        ).length
                                    }{" "}
                                    claim(s) require multi-factor
                                    authentication. Click on the claim to
                                    complete verification.
                                </p>
                            </div>
                        </div>
                    </Card>
                )}

                {/* Filters */}
                <div className="flex items-center gap-4 mb-6">
                    <div className="flex items-center gap-2 text-muted">
                        <Filter className="w-5 h-5" />
                        <span className="font-medium">Filter:</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {[
                            {
                                value: "all",
                                label: "All",
                                count: claims.length,
                            },
                            {
                                value: "pending",
                                label: "Pending",
                                count: statusCounts.pending || 0,
                            },
                            {
                                value: "approved",
                                label: "Approved",
                                count: statusCounts.approved || 0,
                            },
                            {
                                value: "rejected",
                                label: "Rejected",
                                count: statusCounts.rejected || 0,
                            },
                        ].map((filter) => (
                            <button
                                key={filter.value}
                                onClick={() => setFilterStatus(filter.value)}
                                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                                    filterStatus === filter.value
                                        ? "bg-accent text-white"
                                        : "bg-surface text-muted hover:bg-aux"
                                }`}
                            >
                                {filter.label} ({filter.count})
                            </button>
                        ))}
                    </div>
                </div>

                {/* Claims List */}
                {filteredClaims.length === 0 ? (
                    <EmptyState
                        icon={<FileText className="w-16 h-16" />}
                        title={
                            filterStatus === "all"
                                ? "No claims yet"
                                : `No ${filterStatus} claims`
                        }
                        description={
                            filterStatus === "all"
                                ? "Get started by submitting your first claim"
                                : `You don't have any ${filterStatus} claims at the moment`
                        }
                        action={
                            filterStatus === "all" ? (
                                <Button
                                    variant="primary"
                                    onClick={() => navigate("/claims/submit")}
                                    icon={<Plus className="w-5 h-5" />}
                                >
                                    Submit Your First Claim
                                </Button>
                            ) : (
                                <Button
                                    variant="secondary"
                                    onClick={() => setFilterStatus("all")}
                                >
                                    View All Claims
                                </Button>
                            )
                        }
                    />
                ) : (
                    <div className="space-y-4">
                        {filteredClaims.map((claim) => (
                            <Card
                                key={claim.claim_id}
                                hover
                                className="cursor-pointer"
                                onClick={() =>
                                    navigate(`/claims/${claim.claim_id}`)
                                }
                            >
                                <div className="flex items-start gap-4">
                                    {/* Icon */}
                                    <div className="p-3 rounded-lg bg-accent/10 shrink-0">
                                        <FileText className="w-6 h-6 text-accent" />
                                    </div>

                                    {/* Content */}
                                    <div className="flex-1 min-w-0">
                                        {/* Header */}
                                        <div className="flex items-start justify-between gap-4 mb-2">
                                            <div>
                                                <div className="flex items-center gap-2 mb-1">
                                                    <h3 className="text-lg font-semibold text-primary">
                                                        {claim.claim_number}
                                                    </h3>
                                                    {getStatusBadge(
                                                        claim.claim_status
                                                    )}
                                                    {claim.requires_mfa &&
                                                        claim.claim_status ===
                                                            "pending" && (
                                                            <Badge variant="warning">
                                                                MFA Required
                                                            </Badge>
                                                        )}
                                                </div>
                                                <p className="text-sm text-muted capitalize">
                                                    {claim.claim_type.replace(
                                                        /_/g,
                                                        " "
                                                    )}
                                                </p>
                                            </div>
                                            <ChevronRight className="w-5 h-5 text-muted shrink-0" />
                                        </div>

                                        {/* Details */}
                                        <div className="grid grid-cols-2 gap-4 mb-3">
                                            <div className="flex items-center gap-2 text-sm">
                                                <DollarSign className="w-4 h-4 text-muted" />
                                                <span className="font-semibold text-primary">
                                                    {formatCurrency(
                                                        claim.claim_amount
                                                    )}
                                                </span>
                                            </div>
                                            <div className="flex items-center gap-2 text-sm text-muted">
                                                <Calendar className="w-4 h-4" />
                                                <span>
                                                    {formatDate(
                                                        claim.submitted_at
                                                    )}
                                                </span>
                                            </div>
                                        </div>

                                        {/* Description Preview */}
                                        <p className="text-sm text-muted line-clamp-2">
                                            {claim.claim_description}
                                        </p>
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
