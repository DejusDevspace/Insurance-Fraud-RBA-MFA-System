import React, { useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../../hooks/useAuth";
import { useNotification } from "../../hooks/useNotification";
import { Button } from "../common/Button";
import {
    Shield,
    LayoutDashboard,
    FileText,
    LogOut,
    Menu,
    X,
    User,
} from "lucide-react";

const Navbar: React.FC = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const { user, logout } = useAuth();
    const { showNotification } = useNotification();
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

    const handleLogout = async () => {
        try {
            await logout();
            showNotification("success", "Logged out successfully");
            navigate("/login");
        } catch (error) {
            showNotification("error", "Logout failed");
        }
    };

    const isActive = (path: string) => {
        return location.pathname === path;
    };

    const navLinks = user?.is_admin
        ? [
              {
                  path: "/admin/dashboard",
                  label: "Dashboard",
                  icon: LayoutDashboard,
              },
          ]
        : [
              { path: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
              { path: "/claims/history", label: "My Claims", icon: FileText },
          ];

    return (
        <nav className="bg-surface border-b border-aux sticky top-0 z-40">
            <div className="max-w-7xl mx-auto px-4">
                <div className="flex items-center justify-between h-16">
                    {/* Logo */}
                    <Link
                        to={user?.is_admin ? "/admin/dashboard" : "/dashboard"}
                        className="flex items-center gap-2 group"
                    >
                        <div className="p-2 rounded-lg bg-linear-to-br from-accent to-active">
                            <Shield className="w-5 h-5 text-white" />
                        </div>
                        <span className="font-bold text-primary hidden sm:block group-hover:my-gradient transition-all">
                            Insurance Fraud Detection
                        </span>
                    </Link>

                    {/* Desktop Navigation */}
                    <div className="hidden md:flex items-center gap-1">
                        {navLinks.map((link) => {
                            const Icon = link.icon;
                            return (
                                <Link
                                    key={link.path}
                                    to={link.path}
                                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                                        isActive(link.path)
                                            ? "bg-accent/10 text-accent"
                                            : "text-muted hover:text-primary hover:bg-aux"
                                    }`}
                                >
                                    <Icon className="w-4 h-4" />
                                    <span className="font-medium">
                                        {link.label}
                                    </span>
                                </Link>
                            );
                        })}
                    </div>

                    {/* User Menu */}
                    <div className="hidden md:flex items-center gap-4">
                        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-aux">
                            <User className="w-4 h-4 text-accent" />
                            <span className="text-sm font-medium text-primary">
                                {user?.first_name} {user?.last_name}
                            </span>
                            {user?.is_admin && (
                                <span className="ml-2 px-2 py-0.5 text-xs font-semibold bg-accent/20 text-accent rounded">
                                    Admin
                                </span>
                            )}
                        </div>
                        <Button
                            variant="secondary"
                            size="sm"
                            onClick={handleLogout}
                            className="flex gap-2"
                        >
                            Logout
                            <LogOut className="w-4 h-4" />
                        </Button>
                    </div>

                    {/* Mobile Menu Button */}
                    <button
                        className="md:hidden p-2 rounded-lg hover:bg-aux transition-colors"
                        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                    >
                        {mobileMenuOpen ? (
                            <X className="w-6 h-6 text-primary" />
                        ) : (
                            <Menu className="w-6 h-6 text-primary" />
                        )}
                    </button>
                </div>
            </div>

            {/* Mobile Menu */}
            {mobileMenuOpen && (
                <div className="md:hidden border-t border-aux bg-surface">
                    <div className="px-4 py-4 space-y-2">
                        {/* User Info */}
                        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-aux mb-4">
                            <User className="w-4 h-4 text-accent" />
                            <span className="text-sm font-medium text-primary">
                                {user?.first_name} {user?.last_name}
                            </span>
                            {user?.is_admin && (
                                <span className="ml-auto px-2 py-0.5 text-xs font-semibold bg-accent/20 text-accent rounded">
                                    Admin
                                </span>
                            )}
                        </div>

                        {/* Nav Links */}
                        {navLinks.map((link) => {
                            const Icon = link.icon;
                            return (
                                <Link
                                    key={link.path}
                                    to={link.path}
                                    onClick={() => setMobileMenuOpen(false)}
                                    className={`flex items-center gap-2 px-4 py-3 rounded-lg transition-colors ${
                                        isActive(link.path)
                                            ? "bg-accent/10 text-accent"
                                            : "text-muted hover:text-primary hover:bg-aux"
                                    }`}
                                >
                                    <Icon className="w-4 h-4" />
                                    <span className="font-medium">
                                        {link.label}
                                    </span>
                                </Link>
                            );
                        })}

                        {/* Logout */}
                        <button
                            onClick={handleLogout}
                            className="w-full flex items-center gap-2 px-4 py-3 rounded-lg text-muted hover:text-error hover:bg-error/10 transition-colors"
                        >
                            <LogOut className="w-4 h-4" />
                            <span className="font-medium">Logout</span>
                        </button>
                    </div>
                </div>
            )}
        </nav>
    );
};

export default Navbar;
