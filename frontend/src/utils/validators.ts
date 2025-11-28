/**
 * Validate email format
 */
export const validateEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
};

/**
 * Validate password strength
 */
export const validatePassword = (
    password: string
): {
    isValid: boolean;
    errors: string[];
} => {
    const errors: string[] = [];

    if (password.length < 8) {
        errors.push("Password must be at least 8 characters long");
    }

    if (!/\d/.test(password)) {
        errors.push("Password must contain at least one digit");
    }

    if (!/[a-z]/.test(password)) {
        errors.push("Password must contain at least one lowercase letter");
    }

    if (!/[A-Z]/.test(password)) {
        errors.push("Password must contain at least one uppercase letter");
    }

    return {
        isValid: errors.length === 0,
        errors,
    };
};

/**
 * Validate phone number format
 */
export const validatePhoneNumber = (phone: string): boolean => {
    const phoneRegex = /^\+?[1-9]\d{9,14}$/;
    return phoneRegex.test(phone.replace(/[\s\-\(\)\.]/g, ""));
};

/**
 * Validate claim amount
 */
export const validateClaimAmount = (
    amount: number,
    min: number = 1,
    max: number = 1000000
): {
    isValid: boolean;
    error?: string;
} => {
    if (amount < min) {
        return {
            isValid: false,
            error: `Amount must be at least $${min}`,
        };
    }

    if (amount > max) {
        return {
            isValid: false,
            error: `Amount cannot exceed $${max.toLocaleString()}`,
        };
    }

    return { isValid: true };
};

/**
 * Validate date is not in future
 */
export const validateDateNotFuture = (date: Date): boolean => {
    return date <= new Date();
};

/**
 * Validate date is not too old (default: 2 years)
 */
export const validateDateNotTooOld = (
    date: Date,
    maxAgeDays: number = 730
): boolean => {
    const maxDate = new Date();
    maxDate.setDate(maxDate.getDate() - maxAgeDays);
    return date >= maxDate;
};
