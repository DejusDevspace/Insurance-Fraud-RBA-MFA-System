import { forwardRef } from "react";
import type { SelectHTMLAttributes } from "react";
import clsx from "clsx";

interface SelectOption {
    value: string;
    label: string;
}

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
    label?: string;
    error?: string;
    options: SelectOption[];
    placeholder?: string;
    icon?: React.ReactNode;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
    ({ label, error, options, placeholder, icon, className, ...props }, ref) => {
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
                <div className="relative">
                    {icon && (
                        <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted pointer-events-none">
                            {icon}
                        </div>
                    )}
                    <select
                        ref={ref}
                        className={clsx(
                            "input-field cursor-pointer p-2.5",
                            icon && "pl-10",
                            error &&
                                "border-error focus:border-error focus:ring-error",
                            className
                        )}
                        {...props}
                    >
                        {placeholder && (
                            <option value="" disabled>
                                {placeholder}
                            </option>
                        )}
                        {options.map((option) => (
                            <option key={option.value} value={option.value}>
                                {option.label}
                            </option>
                        ))}
                    </select>
                </div>
                {error && <p className="mt-1.5 text-sm text-error">{error}</p>}
            </div>
        );
    }
);

Select.displayName = "Select";
