import React from "react";
import clsx from "clsx";

interface LoadingSpinnerProps {
    size?: "sm" | "md" | "lg" | "xl";
    className?: string;
    text?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
    size = "md",
    className,
    text,
}) => {
    const sizeClasses = {
        sm: "w-4 h-4",
        md: "w-8 h-8",
        lg: "w-12 h-12",
        xl: "w-16 h-16",
    };

    return (
        <div
            className={clsx(
                "flex flex-col items-center justify-center gap-3",
                className
            )}
        >
            <div className={clsx("spinner", sizeClasses[size])} />
            {text && <p className="text-sm text-muted">{text}</p>}
        </div>
    );
};

export const LoadingOverlay: React.FC<{ text?: string }> = ({ text }) => {
    return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
            <div className="card p-8">
                <LoadingSpinner size="lg" text={text || "Loading..."} />
            </div>
        </div>
    );
};
