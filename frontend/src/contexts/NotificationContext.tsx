import React, { createContext, useContext, useState, useCallback } from "react";
import type { ReactNode } from "react";

type NotificationType = "success" | "error" | "warning" | "info";

interface Notification {
    id: string;
    type: NotificationType;
    message: string;
    duration?: number;
}

interface NotificationContextType {
    notifications: Notification[];
    showNotification: (
        type: NotificationType,
        message: string,
        duration?: number
    ) => void;
    removeNotification: (id: string) => void;
}

const NotificationContext = createContext<NotificationContextType | undefined>(
    undefined
);

interface NotificationProviderProps {
    children: ReactNode;
}

export const NotificationProvider: React.FC<NotificationProviderProps> = ({
    children,
}) => {
    const [notifications, setNotifications] = useState<Notification[]>([]);

    const showNotification = useCallback(
        (type: NotificationType, message: string, duration: number = 5000) => {
            const id = Math.random().toString(36).substring(2, 9);
            const notification: Notification = { id, type, message, duration };

            setNotifications((prev) => [...prev, notification]);

            // Auto-remove after duration
            if (duration > 0) {
                setTimeout(() => {
                    removeNotification(id);
                }, duration);
            }
        },
        []
    );

    const removeNotification = useCallback((id: string) => {
        setNotifications((prev) => prev.filter((notif) => notif.id !== id));
    }, []);

    return (
        <NotificationContext.Provider
            value={{
                notifications,
                showNotification,
                removeNotification,
            }}
        >
            {children}
        </NotificationContext.Provider>
    );
};

export const useNotification = () => {
    const context = useContext(NotificationContext);
    if (context === undefined) {
        throw new Error(
            "useNotification must be used within a NotificationProvider"
        );
    }
    return context;
};
