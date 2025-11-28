import React from "react";
import type { HTMLAttributes } from "react";
import clsx from "clsx";

type BadgeVariant =
    | "low"
    | "medium"
    | "high"
    | "success"
    | "warning"
    | "error"
    | "info"
    | "default";

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
    variant?: BadgeVariant;
    size?: "sm" | "md" | "lg";
}

export const Badge: React.FC<BadgeProps> = ({
    children,
    variant = "default",
    size = "md",
    className,
    ...props
}) => {
    const variantClasses = {
        low: "badge-low",
        medium: "badge-medium",
        high: "badge-high",
        success: "bg-greenAccent/10 text-greenAccent border border-greenAccent",
        warning: "bg-warning/10 text-warning border border-warning",
        error: "bg-error/10 text-error border border-error",
        info: "bg-accent/10 text-accent border border-accent",
        default: "bg-aux text-neutral border border-aux",
    };

    const sizeClasses = {
        sm: "px-2 py-0.5 text-xs",
        md: "px-3 py-1 text-sm",
        lg: "px-4 py-1.5 text-base",
    };

    return (
        <span
            className={clsx(
                "badge",
                variantClasses[variant],
                sizeClasses[size],
                className
            )}
            {...props}
        >
            {children}
        </span>
    );
};
