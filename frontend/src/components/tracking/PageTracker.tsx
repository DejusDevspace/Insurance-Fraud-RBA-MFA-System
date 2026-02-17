import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import { useSessionTrackingContext } from "../../contexts/SessionTrackingContext";

const PageTracker: React.FC = () => {
    const location = useLocation();
    const { trackPageVisit } = useSessionTrackingContext();

    useEffect(() => {
        // Track page visit when location changes
        trackPageVisit(location.pathname);
    }, [location.pathname, trackPageVisit]);

    // This component doesn't render anything
    return null;
};

export default PageTracker;
