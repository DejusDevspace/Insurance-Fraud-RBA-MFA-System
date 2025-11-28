import React from "react";
import type { HTMLAttributes } from "react";
import clsx from "clsx";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
    hover?: boolean;
}

export const Card: React.FC<CardProps> = ({
    children,
    hover = false,
    className,
    ...props
}) => {
    return (
        <div
            className={clsx(
                "card",
                hover && "hover:shadow-lg hover:scale-[1.02]",
                className
            )}
            {...props}
        >
            {children}
        </div>
    );
};

interface CardHeaderProps extends HTMLAttributes<HTMLDivElement> {}

export const CardHeader: React.FC<CardHeaderProps> = ({
    children,
    className,
    ...props
}) => {
    return (
        <div className={clsx("mb-4", className)} {...props}>
            {children}
        </div>
    );
};

interface CardTitleProps extends HTMLAttributes<HTMLHeadingElement> {}

export const CardTitle: React.FC<CardTitleProps> = ({
    children,
    className,
    ...props
}) => {
    return (
        <h3
            className={clsx("text-xl font-semibold text-primary", className)}
            {...props}
        >
            {children}
        </h3>
    );
};

interface CardContentProps extends HTMLAttributes<HTMLDivElement> {}

export const CardContent: React.FC<CardContentProps> = ({
    children,
    className,
    ...props
}) => {
    return (
        <div className={clsx("text-primary", className)} {...props}>
            {children}
        </div>
    );
};

interface CardFooterProps extends HTMLAttributes<HTMLDivElement> {}

export const CardFooter: React.FC<CardFooterProps> = ({
    children,
    className,
    ...props
}) => {
    return (
        <div
            className={clsx("mt-4 pt-4 border-t border-aux", className)}
            {...props}
        >
            {children}
        </div>
    );
};
