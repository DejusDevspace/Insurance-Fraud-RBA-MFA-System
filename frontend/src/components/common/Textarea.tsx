import { forwardRef } from "react";
import type { TextareaHTMLAttributes } from "react";
import clsx from "clsx";

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
    label?: string;
    error?: string;
    helperText?: string;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
    ({ label, error, helperText, className, ...props }, ref) => {
        return (
            <div className="w-full">
                {label && (
                    <label className="block text-sm font-medium text-primary mb-1.5">
                        {label}
                        {props.required && (
                            <span className="text-error ml-1">*</span>
                        )}
                    </label>
                )}
                <textarea
                    ref={ref}
                    className={clsx(
                        "input-field resize-none",
                        error &&
                            "border-error focus:border-error focus:ring-error",
                        className
                    )}
                    rows={4}
                    {...props}
                />
                {error && <p className="mt-1.5 text-sm text-error">{error}</p>}
                {helperText && !error && (
                    <p className="mt-1.5 text-sm text-muted">{helperText}</p>
                )}
            </div>
        );
    }
);

Textarea.displayName = "Textarea";
