import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import { NotificationProvider } from "./contexts/NotificationContext";
import LoginPage from "./pages/auth/LoginPage";
import RegisterPage from "./pages/auth/RegisterPage";
import "./App.css";

function App() {
    return (
        <BrowserRouter>
            <AuthProvider>
                <NotificationProvider>
                    <Routes>
                        <Route path="/login" element={<LoginPage />} />
                        <Route path="/register" element={<RegisterPage />} />

                        {/* Redirects */}
                        <Route
                            path="/"
                            element={<Navigate to="/login" replace />}
                        />
                        <Route
                            path="*"
                            element={<Navigate to="/login" replace />}
                        />
                    </Routes>
                </NotificationProvider>
            </AuthProvider>
        </BrowserRouter>
    );
}

export default App;
