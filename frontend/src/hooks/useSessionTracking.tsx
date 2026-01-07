import { useState, useEffect, useCallback, useRef } from "react";
import { useAuth } from "./useAuth";
import { useLocation } from "react-router-dom";

interface SessionTrackingData {
    session_duration: number; // in seconds
    pages_visited: number;
    session_start_time: number;
}

export const useSessionTracking = () => {
    const { isAuthenticated } = useAuth();
    const location = useLocation();
    const [sessionData, setSessionData] = useState<SessionTrackingData>({
        session_duration: 0,
        pages_visited: 1,
        session_start_time: Date.now(),
    });

    const pageCountRef = useRef<Set<string>>(new Set());
    const sessionStartRef = useRef<number>(Date.now());
    const isInitializedRef = useRef<boolean>(false);

    // Initialize session tracking when user logs in
    useEffect(() => {
        if (isAuthenticated && !isInitializedRef.current) {
            sessionStartRef.current = Date.now();
            pageCountRef.current = new Set([window.location.pathname]);
            setSessionData({
                session_duration: 0,
                pages_visited: 1,
                session_start_time: Date.now(),
            });
            isInitializedRef.current = true;
        } else if (!isAuthenticated) {
            isInitializedRef.current = false;
            pageCountRef.current = new Set();
        }
    }, [isAuthenticated]);

    // Track page visits when location changes
    useEffect(() => {
        if (!isAuthenticated || !isInitializedRef.current) return;

        const currentPath = location.pathname;
        console.log("Page change detected!");
        if (!pageCountRef.current.has(currentPath)) {
            pageCountRef.current.add(currentPath);
            setSessionData((prev) => ({
                ...prev,
                pages_visited: pageCountRef.current.size,
            }));
        }
    }, [location.pathname, isAuthenticated]);

    // Update session duration every second
    useEffect(() => {
        if (!isAuthenticated || !isInitializedRef.current) return;

        const interval = setInterval(() => {
            setSessionData((prev) => ({
                ...prev,
                session_duration: Math.floor(
                    (Date.now() - sessionStartRef.current) / 1000
                ),
            }));
        }, 1000);

        return () => clearInterval(interval);
    }, [isAuthenticated]);

    const trackPageVisit = useCallback((path: string) => {
        if (!pageCountRef.current.has(path)) {
            pageCountRef.current.add(path);
            setSessionData((prev) => ({
                ...prev,
                pages_visited: pageCountRef.current.size,
            }));
        }
    }, []);

    return {
        session_duration: sessionData.session_duration,
        pages_visited: sessionData.pages_visited,
        trackPageVisit,
    };
};
