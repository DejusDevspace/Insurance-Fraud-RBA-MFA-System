import React, { useState, useEffect } from "react";
import { useNotification } from "../../hooks/useNotification";
import { mfaService } from "../../services/mfaService";
import { Button } from "../common/Button";
import { Input } from "../common/Input";
import { Shield, Clock, RefreshCw } from "lucide-react";

interface OTPVerificationProps {
    claimId: string;
    onSuccess: () => void;
    onCancel: () => void;
}

const OTPVerification: React.FC<OTPVerificationProps> = ({
    claimId,
    onSuccess,
    onCancel,
}) => {
    const { showNotification } = useNotification();

    const [otpCode, setOtpCode] = useState("");
    const [loading, setLoading] = useState(false);
    const [sendingOTP, setSendingOTP] = useState(false);
    const [otpSent, setOtpSent] = useState(false);
    const [demoOTP, setDemoOTP] = useState<string | null>(null);
    const [timeLeft, setTimeLeft] = useState(300); // 5 minutes

    useEffect(() => {
        sendOTP();
    }, []);

    useEffect(() => {
        if (!otpSent || timeLeft <= 0) return;

        const timer = setInterval(() => {
            setTimeLeft((prev) => {
                if (prev <= 1) {
                    clearInterval(timer);
                    return 0;
                }
                return prev - 1;
            });
        }, 1000);

        return () => clearInterval(timer);
    }, [otpSent, timeLeft]);

    const sendOTP = async () => {
        setSendingOTP(true);
        try {
            const response = await mfaService.sendOTP(claimId);
            setOtpSent(true);
            setTimeLeft(response.expires_in_minutes * 60);

            // Demo mode - show OTP
            if (response.otp_demo) {
                setDemoOTP(response.otp_demo);
                showNotification(
                    "success",
                    `Demo OTP sent: ${response.otp_demo}`
                );
            } else {
                showNotification("success", response.message);
            }
        } catch (error: any) {
            showNotification(
                error.response?.data?.detail || "Failed to send OTP",
                "error"
            );
        } finally {
            setSendingOTP(false);
        }
    };

    const handleVerify = async () => {
        if (!otpCode || otpCode.length !== 6) {
            showNotification("error", "Please enter a valid 6-digit OTP code");
            return;
        }

        setLoading(true);

        try {
            const response = await mfaService.verifyOTP(claimId, otpCode);
            showNotification("success", response.message);
            onSuccess();
        } catch (error: any) {
            showNotification(
                error.response?.data?.detail || "Invalid OTP code",
                "error"
            );
        } finally {
            setLoading(false);
        }
    };

    const formatTime = (seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, "0")}`;
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-accent/10 mb-4">
                    <Shield className="w-8 h-8 text-accent" />
                </div>
                <h3 className="text-xl font-semibold text-primary mb-2">
                    OTP Verification
                </h3>
                <p className="text-sm text-muted">
                    A 6-digit code has been sent to your registered email and
                    phone
                </p>
            </div>

            {/* Demo OTP Display */}
            {demoOTP && (
                <div className="p-4 rounded-lg bg-accent/10 border border-accent">
                    <p className="text-sm font-medium text-accent mb-1">
                        Demo Mode - Your OTP Code:
                    </p>
                    <p className="text-3xl font-bold text-center text-accent tracking-widest">
                        {demoOTP}
                    </p>
                </div>
            )}

            {/* Timer */}
            {otpSent && timeLeft > 0 && (
                <div className="flex items-center justify-center gap-2 text-muted">
                    <Clock className="w-4 h-4" />
                    <span className="text-sm">
                        Code expires in {formatTime(timeLeft)}
                    </span>
                </div>
            )}

            {/* OTP Input */}
            <Input
                label="Enter OTP Code"
                type="text"
                value={otpCode}
                onChange={(e) => {
                    const value = e.target.value.replace(/\D/g, "").slice(0, 6);
                    setOtpCode(value);
                }}
                placeholder="000000"
                disabled={loading || sendingOTP || timeLeft <= 0}
                maxLength={6}
                className="text-center text-2xl tracking-widest font-mono"
            />

            {/* Resend OTP */}
            {timeLeft <= 0 && (
                <div className="text-center">
                    <p className="text-sm text-error mb-3">
                        OTP code has expired
                    </p>
                    <Button
                        variant="secondary"
                        size="sm"
                        onClick={sendOTP}
                        isLoading={sendingOTP}
                        // icon={<RefreshCw className="w-4 h-4" />}
                    >
                        Resend OTP**
                        <RefreshCw className="w-4 h-4" />*
                    </Button>
                </div>
            )}

            {/* Actions */}
            <div className="flex gap-3 pt-4">
                <Button
                    variant="secondary"
                    onClick={onCancel}
                    disabled={loading}
                    className="flex-1"
                >
                    Cancel
                </Button>
                <Button
                    variant="primary"
                    onClick={handleVerify}
                    isLoading={loading}
                    disabled={otpCode.length !== 6 || timeLeft <= 0}
                    className="flex-1"
                >
                    Verify OTP
                </Button>
            </div>

            {/* Help Text */}
            <div className="text-center">
                <p className="text-xs text-muted">
                    Didn't receive the code?{" "}
                    {timeLeft > 0 ? (
                        <span>
                            Please wait {formatTime(timeLeft)} before requesting
                            a new one
                        </span>
                    ) : (
                        <button
                            onClick={sendOTP}
                            disabled={sendingOTP}
                            className="text-accent hover:text-active font-medium"
                        >
                            Resend now
                        </button>
                    )}
                </p>
            </div>
        </div>
    );
};

export default OTPVerification;
