import { forwardRef } from "react";
import type { InputHTMLAttributes } from "react";
import clsx from "clsx";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
    label?: string;
    error?: string;
    helperText?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
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
                <input
                    ref={ref}
                    className={clsx(
                        "input-field",
                        error &&
                            "border-error focus:border-error focus:ring-error",
                        className
                    )}
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

Input.displayName = "Input";
