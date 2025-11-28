import { useState, useCallback } from "react";
import type {
    Claim,
    ClaimSubmission,
    ClaimSubmissionResponse,
} from "../types/claim.types";
import { claimService } from "../services/claimService";
import { handleApiError } from "../services/api";
import { useNotification } from "../hooks/useNotification";

export const useClaims = () => {
    const [claims, setClaims] = useState<Claim[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const { showNotification } = useNotification();

    const fetchClaims = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const data = await claimService.getUserClaims();
            setClaims(data);
        } catch (err) {
            const errorMsg = handleApiError(err);
            setError(errorMsg);
            showNotification("error", errorMsg);
        } finally {
            setIsLoading(false);
        }
    }, [showNotification]);

    const submitClaim = useCallback(
        async (
            claimData: ClaimSubmission
        ): Promise<ClaimSubmissionResponse | null> => {
            setIsLoading(true);
            setError(null);
            try {
                const response = await claimService.submitClaim(claimData);
                showNotification("success", "Claim submitted successfully");
                return response;
            } catch (err) {
                const errorMsg = handleApiError(err);
                setError(errorMsg);
                showNotification("error", errorMsg);
                return null;
            } finally {
                setIsLoading(false);
            }
        },
        [showNotification]
    );

    const getClaimById = useCallback(
        async (claimId: string): Promise<Claim | null> => {
            setIsLoading(true);
            setError(null);
            try {
                const claim = await claimService.getClaimById(claimId);
                return claim;
            } catch (err) {
                const errorMsg = handleApiError(err);
                setError(errorMsg);
                showNotification("error", errorMsg);
                return null;
            } finally {
                setIsLoading(false);
            }
        },
        [showNotification]
    );

    return {
        claims,
        isLoading,
        error,
        fetchClaims,
        submitClaim,
        getClaimById,
    };
};
