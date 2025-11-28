import React from "react";
import type { LucideIcon } from "lucide-react";

interface EmptyStateProps {
    icon: LucideIcon;
    title: string;
    description: string;
    action?: {
        label: string;
        onClick: () => void;
    };
}

export const EmptyState: React.FC<EmptyStateProps> = ({
    icon: Icon,
    title,
    description,
    action,
}) => {
    return (
        <div className="flex flex-col items-center justify-center py-12 px-4">
            <div className="w-16 h-16 rounded-full bg-aux/50 flex items-center justify-center mb-4">
                <Icon className="w-8 h-8 text-neutral" />
            </div>
            <h3 className="text-xl font-semibold text-primary mb-2">{title}</h3>
            <p className="text-muted text-center max-w-md mb-6">
                {description}
            </p>
            {action && (
                <button onClick={action.onClick} className="btn-primary">
                    {action.label}
                </button>
            )}
        </div>
    );
};
