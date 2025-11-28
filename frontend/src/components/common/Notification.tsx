import React, { useEffect } from "react";
import { Alert } from "./Alert";
import { useNotification } from "../../hooks/useNotification";

export const NotificationContainer: React.FC = () => {
    const { notifications, removeNotification } = useNotification();

    return (
        <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-md">
            {notifications.map((notification) => (
                <NotificationItem
                    key={notification.id}
                    notification={notification}
                    onClose={() => removeNotification(notification.id)}
                />
            ))}
        </div>
    );
};

interface NotificationItemProps {
    notification: {
        id: string;
        type: "success" | "error" | "warning" | "info";
        message: string;
    };
    onClose: () => void;
}

const NotificationItem: React.FC<NotificationItemProps> = ({
    notification,
    onClose,
}) => {
    useEffect(() => {
        const timer = setTimeout(() => {
            onClose();
        }, 5000);

        return () => clearTimeout(timer);
    }, [onClose]);

    return (
        <div className="animate-slide-in-right">
            <Alert
                type={notification.type}
                message={notification.message}
                onClose={onClose}
            />
        </div>
    );
};
