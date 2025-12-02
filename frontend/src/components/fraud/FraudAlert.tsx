import React, { useState } from "react";
import { Card } from "../common/Card";
import { Badge } from "../common/Badge";
import { Button } from "../common/Button";
import { LoadingSpinner } from "../common/LoadingSpinner";
import { useNotification } from "../../hooks/useNotification";
import { fraudService } from "../../services/fraudService";
import {
    AlertTriangle,
    Shield,
    ChevronDown,
    ChevronUp,
    TrendingUp,
} from "lucide-react";
import type { FraudDetection, FraudExplanation } from "../../types/fraud.types";

interface FraudAlertProps {
    claimId: string;
    fraudDetection: FraudDetection;
}

const FraudAlert: React.FC<FraudAlertProps> = ({ claimId, fraudDetection }) => {
    const { showNotification } = useNotification();
    const [showDetails, setShowDetails] = useState(false);
    const [explanation, setExplanation] = useState<FraudExplanation | null>(
        null
    );
    const [loadingExplanation, setLoadingExplanation] = useState(false);

    const fetchExplanation = async () => {
        setLoadingExplanation(true);
        try {
            const data = await fraudService.getFraudExplanation(claimId);
            setExplanation(data);
        } catch (error: any) {
            showNotification("error", "Failed to load fraud explanation");
        } finally {
            setLoadingExplanation(false);
        }
    };

    const handleToggleDetails = () => {
        if (!showDetails && !explanation) {
            fetchExplanation();
        }
        setShowDetails(!showDetails);
    };

    const fraudPercentage = Math.round(fraudDetection.fraud_probability * 100);

    const getConfidenceBadge = () => {
        if (fraudPercentage >= 70)
            return <Badge variant="error">High Confidence</Badge>;
        if (fraudPercentage >= 50)
            return <Badge variant="warning">Medium Confidence</Badge>;
        return <Badge variant="info">Low Confidence</Badge>;
    };

    return (
        <Card className="border-error">
            <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                    <div className="p-3 rounded-lg bg-error/10">
                        <AlertTriangle className="w-6 h-6 text-error" />
                    </div>
                    <div>
                        <h2 className="text-xl font-semibold text-primary mb-1">
                            Fraud Detection Alert
                        </h2>
                        <p className="text-sm text-muted">
                            ML Model: {fraudDetection.model_used}
                        </p>
                    </div>
                </div>
                {getConfidenceBadge()}
            </div>

            {/* Warning Message */}
            <div className="p-4 rounded-lg bg-error/10 border border-error mb-6">
                <div className="flex items-start gap-3">
                    <Shield className="w-5 h-5 text-error shrink-0 mt-0.5" />
                    <div>
                        <h3 className="font-semibold text-error mb-1">
                            Suspicious Activity Detected
                        </h3>
                        <p className="text-sm text-muted">
                            This claim has been flagged by our fraud detection
                            system. Please review the analysis below.
                        </p>
                    </div>
                </div>
            </div>

            {/* Fraud Probability */}
            <div className="mb-6">
                <div className="flex items-end justify-between mb-2">
                    <span className="text-sm font-medium text-muted">
                        Fraud Probability
                    </span>
                    <span className="text-3xl font-bold text-error">
                        {fraudPercentage}%
                    </span>
                </div>
                <div className="w-full h-3 bg-surface rounded-full overflow-hidden">
                    <div
                        className="h-full bg-linear-to-r from-error to-red-600 transition-all duration-500"
                        style={{ width: `${fraudPercentage}%` }}
                    />
                </div>
            </div>

            {/* Fraud Type */}
            {fraudDetection.predicted_fraud_type && (
                <div className="mb-4 p-3 rounded-lg bg-surface">
                    <div className="flex items-center justify-between">
                        <span className="text-sm text-muted">
                            Predicted Fraud Type
                        </span>
                        <span className="font-semibold text-error capitalize">
                            {fraudDetection.predicted_fraud_type.replace(
                                /_/g,
                                " "
                            )}
                        </span>
                    </div>
                </div>
            )}

            {/* Anomaly Score */}
            {fraudDetection.anomaly_score !== undefined && (
                <div className="mb-4 p-3 rounded-lg bg-surface">
                    <div className="flex items-center justify-between">
                        <span className="text-sm text-muted">
                            Anomaly Score
                        </span>
                        <span className="font-semibold text-primary">
                            {fraudDetection.anomaly_score.toFixed(3)}
                        </span>
                    </div>
                </div>
            )}

            {/* Toggle Details Button */}
            <Button
                variant="secondary"
                size="sm"
                fullWidth
                onClick={handleToggleDetails}
                icon={
                    showDetails ? (
                        <ChevronUp className="w-4 h-4" />
                    ) : (
                        <ChevronDown className="w-4 h-4" />
                    )
                }
            >
                {showDetails ? "Hide" : "Show"} SHAP Analysis
            </Button>

            {/* Detailed SHAP Explanation */}
            {showDetails && (
                <div className="mt-4 pt-4 border-t border-aux">
                    {loadingExplanation ? (
                        <div className="flex items-center justify-center py-8">
                            <LoadingSpinner size="md" />
                        </div>
                    ) : explanation ? (
                        <div className="space-y-4">
                            {/* Explanation Text */}
                            <div className="p-4 rounded-lg bg-surface">
                                <div className="flex items-start gap-3">
                                    <AlertTriangle className="w-5 h-5 text-error shrink-0 mt-0.5" />
                                    <div>
                                        <h4 className="font-semibold text-primary mb-2">
                                            Fraud Analysis
                                        </h4>
                                        <p className="text-sm text-muted">
                                            {explanation.explanation}
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* Confidence Level */}
                            <div className="p-3 rounded-lg bg-error/10 border border-error">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium text-primary">
                                        Confidence Level
                                    </span>
                                    <span className="text-sm font-semibold text-error">
                                        {explanation.confidence_level}
                                    </span>
                                </div>
                            </div>

                            {/* Top SHAP Features */}
                            {explanation.top_features &&
                                explanation.top_features.length > 0 && (
                                    <div>
                                        <h4 className="font-semibold text-primary mb-3 text-sm">
                                            Key Contributing Features (SHAP
                                            Values)
                                        </h4>
                                        <div className="space-y-2">
                                            {explanation.top_features.map(
                                                (feature, index) => (
                                                    <div
                                                        key={index}
                                                        className="flex items-start gap-3 p-3 rounded-lg bg-surface border border-aux"
                                                    >
                                                        <span className="flex items-center justify-center w-6 h-6 rounded-full bg-error/10 text-error text-xs font-bold shrink-0">
                                                            {index + 1}
                                                        </span>
                                                        <div className="flex-1">
                                                            <div className="flex items-center justify-between mb-1">
                                                                <p className="text-sm font-medium text-primary capitalize">
                                                                    {feature.feature.replace(
                                                                        /_/g,
                                                                        " "
                                                                    )}
                                                                </p>
                                                                <div className="flex items-center gap-1">
                                                                    <TrendingUp className="w-3 h-3 text-error" />
                                                                    <span className="text-xs font-semibold text-error">
                                                                        {(
                                                                            feature.magnitude *
                                                                            100
                                                                        ).toFixed(
                                                                            1
                                                                        )}
                                                                        %
                                                                    </span>
                                                                </div>
                                                            </div>
                                                            <p className="text-xs text-muted">
                                                                SHAP Value:{" "}
                                                                {feature.shap_value.toFixed(
                                                                    4
                                                                )}{" "}
                                                                •{" "}
                                                                {feature.contribution ===
                                                                "increases"
                                                                    ? "Increases"
                                                                    : "Decreases"}{" "}
                                                                fraud
                                                                probability
                                                            </p>
                                                        </div>
                                                    </div>
                                                )
                                            )}
                                        </div>
                                    </div>
                                )}

                            {/* Base Value Info */}
                            {explanation.base_value !== undefined && (
                                <div className="text-xs text-muted text-center p-2 bg-surface rounded">
                                    Model Base Value:{" "}
                                    {explanation.base_value.toFixed(4)}
                                </div>
                            )}
                        </div>
                    ) : (
                        <p className="text-sm text-muted text-center py-4">
                            No detailed explanation available
                        </p>
                    )}
                </div>
            )}
        </Card>
    );
};

export default FraudAlert;
