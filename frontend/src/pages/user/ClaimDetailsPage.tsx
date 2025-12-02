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
    Calendar,
    DollarSign,
    AlertTriangle,
    CheckCircle,
    XCircle,
    ArrowLeft,
    Shield,
    Info,
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
        null
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
                riskService.getRiskScore(claimId!).catch(() => null),
                fraudService.getFraudDetection(claimId!).catch(() => null),
            ]);

            setClaim(claimData);
            setRiskScore(riskData);
            setFraudDetection(fraudData);
        } catch (error: any) {
            showNotification("error", "Failed to load claim details");
            navigate("/claims/history");
        } finally {
            setLoading(false);
        }
    };

    const handleMFASuccess = () => {
        setShowMFAModal(false);
        showNotification(
            "success",
            "Verification successful! Claim has been processed."
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
            month: "long",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
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

    if (!claim) {
        return (
            <div className="min-h-screen bg-background">
                <Navbar />
                <div className="max-w-7xl mx-auto px-4 py-8">
                    <Card>
                        <div className="text-center py-12">
                            <AlertTriangle className="w-16 h-16 text-error mx-auto mb-4" />
                            <h2 className="text-xl font-semibold text-primary mb-2">
                                Claim Not Found
                            </h2>
                            <p className="text-muted mb-6">
                                The claim you're looking for doesn't exist or
                                has been removed.
                            </p>
                            <Button
                                variant="primary"
                                onClick={() => navigate("/claims/history")}
                            >
                                Back to Claims
                            </Button>
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
                {/* Back Button */}
                <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => navigate("/claims/history")}
                    icon={<ArrowLeft className="w-4 h-4" />}
                    className="mb-6"
                >
                    Back to Claims
                </Button>

                {/* Claim Header */}
                <Card className="mb-6">
                    <div className="flex items-start justify-between mb-6">
                        <div className="flex items-start gap-4">
                            <div className="p-3 rounded-lg bg-accent/10">
                                {getStatusIcon(claim.claim_status)}
                            </div>
                            <div>
                                <h1 className="text-2xl font-bold text-primary mb-2">
                                    {claim.claim_number}
                                </h1>
                                <div className="flex items-center gap-2">
                                    {getStatusBadge(claim.claim_status)}
                                    {claim.requires_mfa &&
                                        claim.claim_status === "pending" && (
                                            <Badge variant="warning">
                                                MFA Required
                                            </Badge>
                                        )}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* MFA Action */}
                    {claim.requires_mfa && claim.claim_status === "pending" && (
                        <div className="p-4 rounded-lg bg-warning/10 border border-warning mb-6">
                            <div className="flex items-start gap-3">
                                <AlertTriangle className="w-5 h-5 text-warning shrink-0 mt-0.5" />
                                <div className="flex-1">
                                    <h3 className="font-semibold text-primary mb-1">
                                        Multi-Factor Authentication Required
                                    </h3>
                                    <p className="text-sm text-muted mb-3">
                                        This claim requires additional
                                        verification to complete processing.
                                        Click below to verify your identity.
                                    </p>
                                    <Button
                                        variant="danger"
                                        size="sm"
                                        onClick={() => setShowMFAModal(true)}
                                        icon={<Shield className="w-4 h-4" />}
                                    >
                                        Complete Verification
                                    </Button>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Claim Details Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        <div>
                            <p className="text-sm text-muted mb-1">
                                Claim Type
                            </p>
                            <p className="font-semibold text-primary capitalize">
                                {claim.claim_type.replace(/_/g, " ")}
                            </p>
                        </div>
                        <div>
                            <p className="text-sm text-muted mb-1">
                                Claim Amount
                            </p>
                            <p className="font-semibold text-primary">
                                {formatCurrency(claim.claim_amount)}
                            </p>
                        </div>
                        <div>
                            <p className="text-sm text-muted mb-1">
                                Incident Date
                            </p>
                            <p className="font-semibold text-primary">
                                {new Date(
                                    claim.incident_date
                                ).toLocaleDateString("en-US", {
                                    year: "numeric",
                                    month: "long",
                                    day: "numeric",
                                })}
                            </p>
                        </div>
                        <div>
                            <p className="text-sm text-muted mb-1">Submitted</p>
                            <p className="font-semibold text-primary">
                                {formatDate(claim.submitted_at)}
                            </p>
                        </div>
                    </div>

                    {/* Description */}
                    <div className="mt-6 pt-6 border-t border-aux">
                        <h3 className="font-semibold text-primary mb-3">
                            Description
                        </h3>
                        <p className="text-muted">{claim.claim_description}</p>
                    </div>

                    {/* Supporting Documents */}
                    <div className="mt-4">
                        <p className="text-sm text-muted">
                            <span className="font-medium text-primary">
                                Supporting Documents:
                            </span>{" "}
                            {claim.supporting_documents_count} file(s)
                        </p>
                    </div>

                    {/* Rejection Reason */}
                    {claim.claim_status === "rejected" &&
                        claim.rejection_reason && (
                            <div className="mt-6 p-4 rounded-lg bg-error/10 border border-error">
                                <div className="flex items-start gap-3">
                                    <XCircle className="w-5 h-5 text-error shrink-0 mt-0.5" />
                                    <div>
                                        <h3 className="font-semibold text-error mb-1">
                                            Rejection Reason
                                        </h3>
                                        <p className="text-sm text-muted">
                                            {claim.rejection_reason}
                                        </p>
                                    </div>
                                </div>
                            </div>
                        )}
                </Card>

                {/* Risk Assessment */}
                {riskScore && (
                    <div className="mb-6">
                        <RiskScoreDisplay
                            claimId={claim.claim_id}
                            riskScore={riskScore}
                        />
                    </div>
                )}

                {/* Fraud Detection */}
                {fraudDetection && fraudDetection.is_suspicious && (
                    <div className="mb-6">
                        <FraudAlert
                            claimId={claim.claim_id}
                            fraudDetection={fraudDetection}
                        />
                    </div>
                )}

                {/* Processing Timeline (Placeholder) */}
                <Card>
                    <h2 className="text-xl font-semibold text-primary mb-4">
                        Processing Timeline
                    </h2>
                    <div className="space-y-4">
                        <div className="flex items-start gap-4">
                            <div className="w-2 h-2 rounded-full bg-accent mt-2"></div>
                            <div className="flex-1">
                                <p className="font-medium text-primary">
                                    Claim Submitted
                                </p>
                                <p className="text-sm text-muted">
                                    {formatDate(claim.submitted_at)}
                                </p>
                            </div>
                        </div>
                        {claim.processed_at && (
                            <div className="flex items-start gap-4">
                                <div className="w-2 h-2 rounded-full bg-success mt-2"></div>
                                <div className="flex-1">
                                    <p className="font-medium text-primary">
                                        Claim Processed
                                    </p>
                                    <p className="text-sm text-muted">
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
                    title="Complete Verification"
                    size="md"
                >
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
                </Modal>
            )}
        </div>
    );
};

export default ClaimDetailsPage;
