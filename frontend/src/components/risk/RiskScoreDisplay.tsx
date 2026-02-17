import React, { useState } from "react";
import { Card } from "../common/Card";
import { Badge } from "../common/Badge";
import { Button } from "../common/Button";
import { LoadingSpinner } from "../common/LoadingSpinner";
import { formatExplanationText } from "../../utils/textFormatter";
import { useNotification } from "../../hooks/useNotification";
import { riskService } from "../../services/riskService";
import {
    Shield,
    TrendingUp,
    TrendingDown,
    AlertTriangle,
    ChevronDown,
    ChevronUp,
} from "lucide-react";
import type { RiskScore, RiskAssessment } from "../../types/risk.types";

interface RiskScoreDisplayProps {
    claimId: string;
    riskScore: RiskScore;
}

const RiskScoreDisplay: React.FC<RiskScoreDisplayProps> = ({
    claimId,
    riskScore,
}) => {
    const { showNotification } = useNotification();
    const [showDetails, setShowDetails] = useState(false);
    const [explanation, setExplanation] = useState<RiskAssessment | null>(null);
    const [loadingExplanation, setLoadingExplanation] = useState(false);

    const getRiskColor = (level: string) => {
        switch (level) {
            case "low":
                return "text-success";
            case "medium":
                return "text-warning";
            case "high":
                return "text-error";
            default:
                return "text-muted";
        }
    };

    const getRiskBadge = (level: string) => {
        const variants: Record<string, "success" | "warning" | "error"> = {
            low: "success",
            medium: "warning",
            high: "error",
        };
        return (
            <Badge variant={variants[level] || "info"}>
                {level.toUpperCase()}
            </Badge>
        );
    };

    const fetchExplanation = async () => {
        setLoadingExplanation(true);
        try {
            const data = await riskService.getRiskExplanation(claimId);
            setExplanation(data);
        } catch (error: any) {
            showNotification("error", "Failed to load risk explanation");
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

    const riskPercentage = Math.round(riskScore.risk_score * 100);

    return (
        <Card>
            <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                    <div className="p-3 rounded-lg bg-accent/10">
                        <Shield className="w-6 h-6 text-accent" />
                    </div>
                    <div>
                        <h2 className="text-xl font-semibold text-primary mb-1">
                            Risk Assessment
                        </h2>
                        <p className="text-sm text-muted">
                            AI-powered risk analysis •{" "}
                            {riskScore.calculation_method}
                        </p>
                    </div>
                </div>
                {getRiskBadge(riskScore.risk_level)}
            </div>

            {/* Risk Score Visualization */}
            <div className="mb-6">
                <div className="flex items-end justify-between mb-2">
                    <span className="text-sm font-medium text-muted">
                        Risk Score
                    </span>
                    <span
                        className={`text-3xl font-bold ${getRiskColor(
                            riskScore.risk_level,
                        )}`}
                    >
                        {riskPercentage}%
                    </span>
                </div>
                <div className="w-full h-3 bg-surface rounded-full overflow-hidden">
                    <div
                        className={`h-full transition-all duration-500 ${
                            riskScore.risk_level === "low"
                                ? "bg-linear-to-r from-success to-greenAccent"
                                : riskScore.risk_level === "medium"
                                  ? "bg-linear-to-r from-warning to-yellow-500"
                                  : "bg-linear-to-r from-error to-red-600"
                        }`}
                        style={{ width: `${riskPercentage}%` }}
                    />
                </div>
                <div className="flex justify-between text-xs text-muted mt-1">
                    <span>Low Risk</span>
                    <span>Medium Risk</span>
                    <span>High Risk</span>
                </div>
            </div>

            {/* Top Risk Factors */}
            {Object.keys(riskScore.factors).length > 0 && (
                <div className="mb-4">
                    <h3 className="font-semibold text-primary mb-3 text-sm">
                        Key Risk Factors
                    </h3>
                    <div className="space-y-2">
                        {Object.entries(riskScore.factors)
                            .sort(
                                ([, a], [, b]) =>
                                    Math.abs(b as number) -
                                    Math.abs(a as number),
                            )
                            .slice(0, 3)
                            .map(([factor, value]) => (
                                <div
                                    key={factor}
                                    className="flex items-center justify-between p-2 rounded-lg bg-surface"
                                >
                                    <span className="text-sm text-muted capitalize">
                                        {factor.replace(/_/g, " ")}
                                    </span>
                                    <div className="flex items-center gap-2">
                                        {(value as number) > 0 ? (
                                            <TrendingUp className="w-4 h-4 text-error" />
                                        ) : (
                                            <TrendingDown className="w-4 h-4 text-success" />
                                        )}
                                        <span className="text-sm font-medium text-primary">
                                            {Math.abs(value as number).toFixed(
                                                2,
                                            )}
                                        </span>
                                    </div>
                                </div>
                            ))}
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
                {showDetails ? "Hide" : "Show"} Detailed Analysis
            </Button>

            {/* Detailed Explanation */}
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
                                    <AlertTriangle className="w-5 h-5 text-accent shrink-0 mt-0.5" />
                                    <div>
                                        <h4 className="font-semibold text-primary mb-2">
                                            Risk Analysis
                                        </h4>
                                        <div className="text-sm text-muted whitespace-pre-line">
                                            {formatExplanationText(
                                                explanation.explanation,
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Top Contributing Factors */}
                            {explanation.top_risk_factors &&
                                explanation.top_risk_factors.length > 0 && (
                                    <div>
                                        <h4 className="font-semibold text-primary mb-3 text-sm">
                                            Contributing Factors
                                        </h4>
                                        <div className="space-y-2">
                                            {explanation.top_risk_factors.map(
                                                (factor, index) => (
                                                    <div
                                                        key={index}
                                                        className="flex items-start gap-3 p-3 rounded-lg bg-surface border border-aux"
                                                    >
                                                        <span className="flex items-center justify-center w-6 h-6 rounded-full bg-accent/10 text-accent text-xs font-bold shrink-0">
                                                            {index + 1}
                                                        </span>
                                                        <div className="flex-1">
                                                            <p className="text-sm font-medium text-primary capitalize">
                                                                {factor.factor.replace(
                                                                    /_/g,
                                                                    " ",
                                                                )}
                                                            </p>
                                                            <p className="text-xs text-muted mt-1">
                                                                Impact:{" "}
                                                                {factor.shap_value >
                                                                0
                                                                    ? "Increases"
                                                                    : "Decreases"}{" "}
                                                                risk by{" "}
                                                                {(
                                                                    Math.abs(
                                                                        factor.magnitude,
                                                                    ) * 100
                                                                ).toFixed(1)}
                                                                %
                                                            </p>
                                                        </div>
                                                    </div>
                                                ),
                                            )}
                                        </div>
                                    </div>
                                )}

                            {/* MFA Recommendation */}
                            {explanation.requires_mfa && (
                                <div className="p-4 rounded-lg bg-warning/10 border border-warning">
                                    <div className="flex items-start gap-3">
                                        <Shield className="w-5 h-5 text-warning shrink-0 mt-0.5" />
                                        <div>
                                            <h4 className="font-semibold text-primary mb-1">
                                                Additional Verification
                                                Recommended
                                            </h4>
                                            <p className="text-sm text-muted">
                                                Based on the risk assessment,{" "}
                                                {explanation.mfa_method?.toUpperCase()}{" "}
                                                verification is required to
                                                process this claim securely.
                                            </p>
                                        </div>
                                    </div>
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

export default RiskScoreDisplay;
