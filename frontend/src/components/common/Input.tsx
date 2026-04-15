import { forwardRef } from "react";
import type { InputHTMLAttributes } from "react";
import clsx from "clsx";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
	label?: string;
	error?: string;
	helperText?: string;
	icon?: React.ReactNode;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
	({ label, error, helperText, icon, className, ...props }, ref) => {
		return (
			<div className="w-full">
				{label && (
					<label className="block text-sm font-medium text-primary mb-1.5">
						{label}
						{props.required && <span className="text-error ml-1">*</span>}
					</label>
				)}
				<div className="relative">
					{icon && (
						<div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted pointer-events-none">
							{icon}
						</div>
					)}
					<input
						ref={ref}
						className={clsx(
							"input-field p-2.5",
							icon && "pl-12!",
							error && "border-error focus:border-error focus:ring-error",
							className,
						)}
						{...props}
					/>
				</div>
				{error && <p className="mt-1.5 text-sm text-error">{error}</p>}
				{helperText && !error && (
					<p className="mt-1.5 text-sm text-muted">{helperText}</p>
				)}
			</div>
		);
	},
);

Input.displayName = "Input";
