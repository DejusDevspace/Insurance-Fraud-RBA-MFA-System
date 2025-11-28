import { useState, useCallback } from "react";
import { handleApiError } from "../services/api";
import { useNotification } from "../hooks/useNotification";

export const useApi = <T, P extends any[] = []>(
    apiFunction: (...args: P) => Promise<T>,
    showSuccessMessage: boolean = false,
    successMessage: string = "Operation completed successfully"
) => {
    const [data, setData] = useState<T | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const { showNotification } = useNotification();

    const execute = useCallback(
        async (...args: P): Promise<T | null> => {
            setIsLoading(true);
            setError(null);
            try {
                const result = await apiFunction(...args);
                setData(result);
                if (showSuccessMessage) {
                    showNotification("success", successMessage);
                }
                return result;
            } catch (err) {
                const errorMsg = handleApiError(err);
                setError(errorMsg);
                showNotification("error", errorMsg);
                return null;
            } finally {
                setIsLoading(false);
            }
        },
        [apiFunction, showNotification, showSuccessMessage, successMessage]
    );

    const reset = useCallback(() => {
        setData(null);
        setError(null);
        setIsLoading(false);
    }, []);

    return {
        data,
        isLoading,
        error,
        execute,
        reset,
    };
};
