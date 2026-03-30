'use client';

import { Suspense } from 'react';
import Navbar from '@/components/Navbar';
import AlertsPanel from '@/components/AlertsPanel';
import { useWebSocket } from '@/lib/websocket';
import { Activity } from 'lucide-react';

export default function AlertsPage() {
    // We reuse the websocket hook to keep the "Connected" UI up-to-date
    // and to push live alerts into the AlertsPanel if new ones arrive natively
    const { isConnected, lastMessage } = useWebSocket('ws://localhost:8000/ws');

    // Extract ws alerts specifically
    const wsAlerts = lastMessage?.type === 'alert' ? [lastMessage.data] : [];

    return (
        <div className="min-h-screen">
            <Navbar isConnected={isConnected} />
            
            <main className="container mx-auto px-4 py-8">
                <div className="mb-6 flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-accent-primary to-accent-secondary">
                            System Alerts History
                        </h1>
                        <p className="text-gray-400 mt-2">
                            Comprehensive chronological log of all danger, traffic, and system flags recorded by the AI engine.
                        </p>
                    </div>
                </div>

                <div className="max-w-5xl mx-auto mt-8">
                    <Suspense fallback={<div className="text-center text-gray-400 py-10">Loading history...</div>}>
                        <AlertsPanel fullPage={true} alerts={wsAlerts} />
                    </Suspense>
                </div>
            </main>
        </div>
    );
}
