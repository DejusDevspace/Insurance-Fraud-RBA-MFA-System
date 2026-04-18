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
		<nav className="fixed top-0 left-0 right-0 z-50 transition-all duration-300 bg-surface/80 backdrop-blur-md border-b border-white/10 shadow-lg shadow-black/20">
			<div className="max-w-7xl mx-auto px-4">
				<div className="flex items-center justify-between h-20">
					{/* Logo */}
					<Link
						to={user?.is_admin ? "/admin/dashboard" : "/dashboard"}
						className="flex items-center gap-3 active:scale-95 transition-transform"
					>
						<div className="w-10 h-10 rounded-xl bg-accent/20 backdrop-blur-md flex items-center justify-center border border-accent/30 shadow-lg shadow-accent/20">
							<Shield className="w-6 h-6 text-accent" />
						</div>
						<span className="text-xl font-bold tracking-tight text-white hidden sm:block">
							Fraud <span className="my-gradient">Protection</span>
						</span>
					</Link>

					{/* Desktop Navigation */}
					<div className="hidden md:flex items-center gap-1.5 p-1.5 bg-background/50 backdrop-blur-sm rounded-xl border border-white/5">
						{navLinks.map((link) => {
							const Icon = link.icon;
							const active = isActive(link.path);
							return (
								<Link
									key={link.path}
									to={link.path}
									className={`flex items-center gap-2 px-5 py-2.5 rounded-lg transition-all duration-300 ${
										active
											? "bg-accent text-white shadow-lg shadow-accent/20"
											: "text-white/60 hover:text-white hover:bg-white/5"
									}`}
								>
									<Icon
										className={`w-4 h-4 ${active ? "animate-pulse" : ""}`}
									/>
									<span className="font-semibold text-sm">{link.label}</span>
								</Link>
							);
						})}
					</div>

					{/* User Menu */}
					<div className="hidden md:flex items-center gap-6">
						<div className="flex items-center gap-3 pr-6 border-r border-white/10">
							<div className="w-10 h-10 rounded-full bg-linear-to-br from-accent/20 to-active/20 flex items-center justify-center border border-accent/10">
								<User className="w-5 h-5 text-accent" />
							</div>
							<div className="flex flex-col">
								<span className="text-sm font-bold text-white leading-none mb-1">
									{user?.first_name} {user?.last_name}
								</span>
								{user?.is_admin ? (
									<span className="text-[10px] uppercase tracking-wider font-black text-accent">
										Admin Panel
									</span>
								) : (
									<span className="text-[10px] uppercase tracking-wider font-black text-white/40">
										Policy Holder
									</span>
								)}
							</div>
						</div>
						<Button
							variant="secondary"
							onClick={handleLogout}
							className="bg-white/5 border-white/10 hover:bg-error/10 hover:border-error/20 hover:text-error transition-all group"
						>
							<LogOut className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
							<span className="text-sm font-bold">Logout</span>
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
									<span className="font-medium">{link.label}</span>
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
