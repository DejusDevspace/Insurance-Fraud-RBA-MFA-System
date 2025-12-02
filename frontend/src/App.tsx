import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import ProtectedRoute from "./components/auth/ProtectedRoute";
import { NotificationProvider } from "./contexts/NotificationContext";
import LoginPage from "./pages/auth/LoginPage";
import RegisterPage from "./pages/auth/RegisterPage";
import DashboardPage from "./pages/user/DashboardPage";
import SubmitClaimPage from "./pages/user/SubmitClaimPage";
import ClaimsHistoryPage from "./pages/user/ClaimsHistoryPage";
import ClaimDetailsPage from "./pages/user/ClaimDetailsPage";

function App() {
    return (
        <BrowserRouter>
            <AuthProvider>
                <NotificationProvider>
                    <Routes>
                        <Route path="/login" element={<LoginPage />} />
                        <Route path="/register" element={<RegisterPage />} />

                        {/* Protected User Routes */}
                        <Route
                            path="/dashboard"
                            element={
                                // <ProtectedRoute>
                                <DashboardPage />
                                // </ProtectedRoute>
                            }
                        />
                        <Route
                            path="/claims/submit"
                            element={
                                // <ProtectedRoute>
                                <SubmitClaimPage />
                                // </ProtectedRoute>
                            }
                        />
                        <Route
                            path="/claims/history"
                            element={
                                // <ProtectedRoute>
                                <ClaimsHistoryPage />
                                // </ProtectedRoute>
                            }
                        />
                        <Route
                            path="/claims/:claimId"
                            element={
                                <ProtectedRoute>
                                    <ClaimDetailsPage />
                                </ProtectedRoute>
                            }
                        />

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
