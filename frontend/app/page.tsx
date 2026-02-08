'use client';

import { useState, useEffect } from 'react';
import Navbar from '@/components/Navbar';
import LiveStream from '@/components/LiveStream';
import AlertsPanel from '@/components/AlertsPanel';
import StatsCards from '@/components/StatsCards';
import VideoUpload from '@/components/VideoUpload';
import { useWebSocket } from '@/lib/websocket';

export default function Dashboard() {
    const [activeTab, setActiveTab] = useState<'live' | 'upload'>('live');
    const [alerts, setAlerts] = useState<any[]>([]);
    const [stats, setStats] = useState({
        totalAlerts: 0,
        activeStreams: 0,
        detections: 0,
        uptime: '0h 0m'
    });

    // WebSocket connection for real-time alerts
    const { isConnected, lastMessage } = useWebSocket('ws://localhost:8000/ws');

    useEffect(() => {
        if (lastMessage?.type === 'alert') {
            setAlerts(prev => [lastMessage.data, ...prev].slice(0, 50));
            setStats(prev => ({ ...prev, totalAlerts: prev.totalAlerts + 1 }));
        }
    }, [lastMessage]);

    // Fetch initial stats
    useEffect(() => {
        fetch('http://localhost:8000/api/alerts/stats')
            .then(res => res.json())
            .then(data => {
                setStats(prev => ({
                    ...prev,
                    totalAlerts: data.total || 0
                }));
            })
            .catch(() => { });
    }, []);

    return (
        <div className="min-h-screen">
            <Navbar isConnected={isConnected} />

            <main className="container mx-auto px-4 py-6">
                {/* Stats Cards */}
                <StatsCards stats={stats} />

                {/* Tab Navigation */}
                <div className="flex gap-4 mb-6 mt-8">
                    <button
                        onClick={() => setActiveTab('live')}
                        className={`px-6 py-3 rounded-xl font-medium transition-all ${activeTab === 'live'
                                ? 'bg-accent-primary text-white glow-accent'
                                : 'glass-card text-gray-400 hover:text-white'
                            }`}
                    >
                        🎥 Live Stream
                    </button>
                    <button
                        onClick={() => setActiveTab('upload')}
                        className={`px-6 py-3 rounded-xl font-medium transition-all ${activeTab === 'upload'
                                ? 'bg-accent-primary text-white glow-accent'
                                : 'glass-card text-gray-400 hover:text-white'
                            }`}
                    >
                        📤 Upload Video
                    </button>
                </div>

                {/* Main Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Video Section */}
                    <div className="lg:col-span-2">
                        {activeTab === 'live' ? (
                            <LiveStream />
                        ) : (
                            <VideoUpload />
                        )}
                    </div>

                    {/* Alerts Panel */}
                    <div className="lg:col-span-1">
                        <AlertsPanel alerts={alerts} />
                    </div>
                </div>
            </main>
        </div>
    );
}
