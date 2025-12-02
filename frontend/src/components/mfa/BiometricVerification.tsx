import React, { useState } from "react";
import { useNotification } from "../../hooks/useNotification";
import { mfaService } from "../../services/mfaService";
import { Button } from "../common/Button";
import { Fingerprint, Shield, CheckCircle } from "lucide-react";

interface BiometricVerificationProps {
    claimId: string;
    onSuccess: () => void;
    onCancel: () => void;
}

const BiometricVerification: React.FC<BiometricVerificationProps> = ({
    claimId,
    onSuccess,
    onCancel,
}) => {
    const { showNotification } = useNotification();

    const [loading, setLoading] = useState(false);
    const [verifying, setVerifying] = useState(false);
    const [verified, setVerified] = useState(false);

    const simulateBiometricScan = (): Promise<boolean> => {
        return new Promise((resolve) => {
            // Simulate biometric scan delay
            setTimeout(() => {
                // 95% success rate for demo
                resolve(Math.random() > 0.05);
            }, 2000);
        });
    };

    const handleVerify = async () => {
        setVerifying(true);

        try {
            // Simulate biometric scan
            const scanSuccess = await simulateBiometricScan();

            if (!scanSuccess) {
                showNotification(
                    "error",
                    "Biometric verification failed. Please try again."
                );
                setVerifying(false);
                return;
            }

            setVerified(true);
            setVerifying(false);

            // Submit to backend
            setLoading(true);
            const response = await mfaService.verifyBiometric(claimId);
            showNotification("success", response.message);

            setTimeout(() => {
                onSuccess();
            }, 1000);
        } catch (error: any) {
            setVerified(false);
            showNotification(
                error.response?.data?.detail || "Biometric verification failed",
                "error"
            );
        } finally {
            setLoading(false);
            setVerifying(false);
        }
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-accent/10 mb-4">
                    <Fingerprint className="w-8 h-8 text-accent" />
                </div>
                <h3 className="text-xl font-semibold text-primary mb-2">
                    Biometric Verification
                </h3>
                <p className="text-sm text-muted">
                    Verify your identity using biometric authentication
                </p>
            </div>

            {/* Verification Status */}
            <div className="relative">
                {/* Scanning Animation */}
                {verifying && (
                    <div className="flex flex-col items-center justify-center py-8">
                        <div className="relative">
                            <Fingerprint className="w-24 h-24 text-accent animate-pulse" />
                            <div className="absolute inset-0 flex items-center justify-center">
                                <div className="w-32 h-32 border-4 border-accent border-t-transparent rounded-full animate-spin" />
                            </div>
                        </div>
                        <p className="text-muted mt-6 animate-pulse">
                            Scanning biometric data...
                        </p>
                    </div>
                )}

                {/* Success State */}
                {verified && !verifying && (
                    <div className="flex flex-col items-center justify-center py-8">
                        <div className="relative">
                            <div className="w-24 h-24 rounded-full bg-success/10 flex items-center justify-center">
                                <CheckCircle className="w-16 h-16 text-success" />
                            </div>
                        </div>
                        <p className="text-success font-semibold mt-6">
                            Verification Successful!
                        </p>
                    </div>
                )}

                {/* Initial State */}
                {!verifying && !verified && (
                    <div className="flex flex-col items-center justify-center py-8">
                        <div className="w-24 h-24 rounded-full bg-accent/10 flex items-center justify-center mb-6">
                            <Fingerprint className="w-16 h-16 text-accent" />
                        </div>
                        <p className="text-muted text-center">
                            Click the button below to start biometric
                            verification
                        </p>
                    </div>
                )}
            </div>

            {/* Info Box */}
            <div className="p-4 rounded-lg bg-surface border border-aux">
                <div className="flex items-start gap-3">
                    <Shield className="w-5 h-5 text-accent shrink-0 mt-0.5" />
                    <div className="text-sm text-muted">
                        <p className="font-medium text-primary mb-1">
                            Demo Mode
                        </p>
                        <p>
                            This is a simulated biometric verification. In
                            production, this would use actual fingerprint,
                            facial recognition, or other biometric sensors.
                        </p>
                    </div>
                </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3 pt-4">
                <Button
                    variant="secondary"
                    onClick={onCancel}
                    disabled={loading || verifying}
                    className="flex-1"
                >
                    Cancel
                </Button>
                <Button
                    variant="primary"
                    onClick={handleVerify}
                    isLoading={loading || verifying}
                    disabled={verified}
                    // icon={<Fingerprint className="w-5 h-5" />}
                    className="flex-1"
                >
                    {verified ? "Verified" : "Verify Biometric"}
                    <Fingerprint className="w-5 h-5" />
                </Button>
            </div>

            {/* Help Text */}
            <div className="text-center">
                <p className="text-xs text-muted">
                    Your biometric data is processed securely and never stored
                </p>
            </div>
        </div>
    );
};

export default BiometricVerification;
