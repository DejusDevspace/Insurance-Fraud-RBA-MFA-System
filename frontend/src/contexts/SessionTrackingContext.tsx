import React, {
    createContext,
    useContext,
    useState,
    useEffect,
    useRef,
    useCallback,
} from "react";
import { useAuth } from "../hooks/useAuth";

interface SessionTrackingData {
    session_start_time: number;
    session_duration: number;
    pages_visited: number;
    is_active: boolean;
}

interface SessionTrackingContextType {
    sessionData: SessionTrackingData;
    startSession: () => void;
    endSession: () => void;
    trackPageVisit: (path: string) => void;
    getSessionDuration: () => number;
}

const SessionTrackingContext = createContext<
    SessionTrackingContextType | undefined
>(undefined);

export const useSessionTrackingContext = () => {
    const context = useContext(SessionTrackingContext);
    if (!context) {
        throw new Error(
            "useSessionTrackingContext must be used within a SessionTrackingProvider",
        );
    }
    return context;
};

export const SessionTrackingProvider: React.FC<{
    children: React.ReactNode;
}> = ({ children }) => {
    const { isAuthenticated } = useAuth();

    // Session state
    const [sessionData, setSessionData] = useState<SessionTrackingData>({
        session_start_time: 0,
        session_duration: 0,
        pages_visited: 0,
        is_active: false,
    });

    // Refs for tracking
    const sessionStartRef = useRef<number>(0);
    const visitedPagesRef = useRef<Set<string>>(new Set());
    const intervalRef = useRef<number | null>(null);
    const isInitializedRef = useRef<boolean>(false);

    // Start session when user authenticates
    useEffect(() => {
        if (isAuthenticated && !isInitializedRef.current) {
            startSession();
            isInitializedRef.current = true;
        } else if (!isAuthenticated && isInitializedRef.current) {
            endSession();
            isInitializedRef.current = false;
        }
    }, [isAuthenticated]);

    // Update session duration every second
    useEffect(() => {
        if (sessionData.is_active && !intervalRef.current) {
            intervalRef.current = setInterval(() => {
                const currentDuration = Math.floor(
                    (Date.now() - sessionStartRef.current) / 1000,
                );
                setSessionData((prev) => ({
                    ...prev,
                    session_duration: currentDuration,
                }));
            }, 1000);
        } else if (!sessionData.is_active && intervalRef.current) {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
        }

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
        };
    }, [sessionData.is_active]);

    const startSession = useCallback(() => {
        const now = Date.now();
        sessionStartRef.current = now;
        visitedPagesRef.current = new Set([window.location.pathname]);

        setSessionData({
            session_start_time: now,
            session_duration: 0,
            pages_visited: 1,
            is_active: true,
        });

        console.log(
            "Session tracking started at:",
            new Date(now).toISOString(),
        );
    }, []);

    const endSession = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }

        const finalDuration = Math.floor(
            (Date.now() - sessionStartRef.current) / 1000,
        );

        setSessionData((prev) => ({
            ...prev,
            session_duration: finalDuration,
            is_active: false,
        }));

        console.log(
            "Session tracking ended. Final duration:",
            finalDuration,
            "seconds",
        );
    }, []);

    const trackPageVisit = useCallback(
        (path: string) => {
            if (!sessionData.is_active) return;

            if (!visitedPagesRef.current.has(path)) {
                visitedPagesRef.current.add(path);
                setSessionData((prev) => ({
                    ...prev,
                    pages_visited: visitedPagesRef.current.size,
                }));

                console.log(
                    "Page visited:",
                    path,
                    "Total pages:",
                    visitedPagesRef.current.size,
                );
            }
        },
        [sessionData.is_active],
    );

    const getSessionDuration = useCallback(() => {
        if (sessionData.is_active) {
            return Math.floor((Date.now() - sessionStartRef.current) / 1000);
        }
        return sessionData.session_duration;
    }, [sessionData.is_active, sessionData.session_duration]);

    const contextValue: SessionTrackingContextType = {
        sessionData,
        startSession,
        endSession,
        trackPageVisit,
        getSessionDuration,
    };

    return (
        <SessionTrackingContext.Provider value={contextValue}>
            {children}
        </SessionTrackingContext.Provider>
    );
};
