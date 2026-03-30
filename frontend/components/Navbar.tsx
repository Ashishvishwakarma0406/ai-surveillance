'use client';

import { Activity, Bell, Settings, Menu } from 'lucide-react';
import Link from 'next/link';

interface NavbarProps {
    isConnected: boolean;
}

export default function Navbar({ isConnected }: NavbarProps) {
    return (
        <nav className="glass-card mx-4 mt-4 px-6 py-4">
            <div className="flex items-center justify-between">
                {/* Logo */}
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-primary to-accent-secondary flex items-center justify-center">
                        <Activity className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold">HawkEye AI</h1>
                        <p className="text-xs text-gray-400">Intelligent Surveillance System</p>
                    </div>
                </div>

                {/* Status & Actions */}
                <div className="flex items-center gap-4">
                    {/* Connection Status */}
                    <div className="flex items-center gap-2 px-3 py-2 rounded-lg glass-card">
                        <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-accent-success animate-pulse' : 'bg-accent-danger'
                            }`} />
                        <span className="text-sm text-gray-300">
                            {isConnected ? 'Connected' : 'Disconnected'}
                        </span>
                    </div>

                    <Link href="/" className="px-3 py-2 hover:bg-dark-600 rounded-lg transition-colors flex items-center gap-2">
                        <Activity className="w-5 h-5 text-gray-400" />
                        <span className="text-sm hidden sm:inline text-gray-300">Dashboard</span>
                    </Link>

                    <Link href="/alerts" className="px-3 py-2 hover:bg-dark-600 rounded-lg transition-colors flex items-center gap-2 bg-dark-700/50">
                        <Bell className="w-5 h-5 text-accent-primary" />
                        <span className="text-sm hidden sm:inline text-white">Full History</span>
                    </Link>
                </div>
            </div>
        </nav>
    );
}
