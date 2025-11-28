import React from "react";
import type { ReactNode } from "react";
import { AlertCircle, CheckCircle, Info, XCircle, X } from "lucide-react";
import clsx from "clsx";

type AlertType = "success" | "error" | "warning" | "info";

interface AlertProps {
    type: AlertType;
    title?: string;
    message: string | ReactNode;
    onClose?: () => void;
    className?: string;
}

export const Alert: React.FC<AlertProps> = ({
    type,
    title,
    message,
    onClose,
    className,
}) => {
    const config = {
        success: {
            icon: CheckCircle,
            bgColor: "bg-greenAccent/10",
            borderColor: "border-greenAccent",
            textColor: "text-greenAccent",
            iconColor: "text-greenAccent",
        },
        error: {
            icon: XCircle,
            bgColor: "bg-error/10",
            borderColor: "border-error",
            textColor: "text-error",
            iconColor: "text-error",
        },
        warning: {
            icon: AlertCircle,
            bgColor: "bg-warning/10",
            borderColor: "border-warning",
            textColor: "text-warning",
            iconColor: "text-warning",
        },
        info: {
            icon: Info,
            bgColor: "bg-accent/10",
            borderColor: "border-accent",
            textColor: "text-accent",
            iconColor: "text-accent",
        },
    };

    const {
        icon: Icon,
        bgColor,
        borderColor,
        textColor,
        iconColor,
    } = config[type];

    return (
        <div
            className={clsx(
                "rounded-lg p-4 border",
                bgColor,
                borderColor,
                className
            )}
        >
            <div className="flex items-start gap-3">
                <Icon className={clsx("w-5 h-5 shrink-0 mt-0.5", iconColor)} />
                <div className="flex-1">
                    {title && (
                        <h4 className={clsx("font-semibold mb-1", textColor)}>
                            {title}
                        </h4>
                    )}
                    <div className={clsx("text-sm", textColor)}>{message}</div>
                </div>
                {onClose && (
                    <button
                        onClick={onClose}
                        className={clsx(
                            "p-1 rounded hover:bg-black/10",
                            textColor
                        )}
                    >
                        <X className="w-4 h-4" />
                    </button>
                )}
            </div>
        </div>
    );
};
