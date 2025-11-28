import { format, formatDistanceToNow, parseISO } from "date-fns";

/**
 * Format currency amount
 */
export const formatCurrency = (amount: number): string => {
    return new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
    }).format(amount);
};

/**
 * Format date to readable string
 */
export const formatDate = (date: string | Date): string => {
    const dateObj = typeof date === "string" ? parseISO(date) : date;
    return format(dateObj, "MMM dd, yyyy");
};

/**
 * Format datetime to readable string
 */
export const formatDateTime = (date: string | Date): string => {
    const dateObj = typeof date === "string" ? parseISO(date) : date;
    return format(dateObj, "MMM dd, yyyy hh:mm a");
};

/**
 * Format date to relative time (e.g., "2 hours ago")
 */
export const formatRelativeTime = (date: string | Date): string => {
    const dateObj = typeof date === "string" ? parseISO(date) : date;
    return formatDistanceToNow(dateObj, { addSuffix: true });
};

/**
 * Format percentage
 */
export const formatPercentage = (
    value: number,
    decimals: number = 1
): string => {
    return `${value.toFixed(decimals)}%`;
};

/**
 * Format risk score (0-1) to percentage
 */
export const formatRiskScore = (score: number): string => {
    return formatPercentage(score * 100);
};

/**
 * Truncate string with ellipsis
 */
export const truncateString = (
    str: string,
    maxLength: number = 100
): string => {
    if (str.length <= maxLength) return str;
    return str.substring(0, maxLength) + "...";
};

/**
 * Capitalize first letter
 */
export const capitalize = (str: string): string => {
    return str.charAt(0).toUpperCase() + str.slice(1);
};

/**
 * Format claim number for display
 */
export const formatClaimNumber = (claimNumber: string): string => {
    return claimNumber.replace("CLM-", "CLM-");
};

/**
 * Format feature name (convert snake_case to Title Case)
 */
export const formatFeatureName = (feature: string): string => {
    return feature
        .split("_")
        .map((word) => capitalize(word))
        .join(" ");
};
