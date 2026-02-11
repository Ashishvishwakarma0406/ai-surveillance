'use client';

import { useState, useEffect } from 'react';
import { AlertTriangle, AlertCircle, Info, Check, RefreshCw } from 'lucide-react';

interface Alert {
    id: number;
    alert_type: string;
    severity: string;
    message: string;
    confidence: number;
    timestamp: string;
    acknowledged: boolean;
}

interface AlertsPanelProps {
    alerts?: Alert[];  // Optional - can receive from parent via WebSocket
}

export default function AlertsPanel({ alerts: wsAlerts = [] }: AlertsPanelProps) {
    const [fetchedAlerts, setFetchedAlerts] = useState<Alert[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Fetch alerts from API on mount
    const fetchAlerts = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const res = await fetch('http://localhost:8000/api/alerts?limit=50');
            if (!res.ok) throw new Error('Failed to fetch alerts');
            const data = await res.json();
            setFetchedAlerts(data);
        } catch (err) {
            setError('Could not load alerts');
            console.error('Fetch alerts error:', err);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchAlerts();
    }, []);

    // Merge WebSocket alerts with fetched alerts (WebSocket takes priority)
    const allAlerts = [...wsAlerts, ...fetchedAlerts.filter(
        fetched => !wsAlerts.some(ws => ws.id === fetched.id)
    )].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

    const getSeverityIcon = (severity: string) => {
        switch (severity) {
            case 'critical':
                return <AlertTriangle className="w-5 h-5 text-accent-danger" />;
            case 'warning':
                return <AlertCircle className="w-5 h-5 text-accent-warning" />;
            default:
                return <Info className="w-5 h-5 text-accent-info" />;
        }
    };

    const getSeverityClass = (severity: string) => {
        switch (severity) {
            case 'critical':
                return 'severity-critical';
            case 'warning':
                return 'severity-warning';
            default:
                return 'severity-info';
        }
    };

    const formatTime = (timestamp: string) => {
        return new Date(timestamp).toLocaleTimeString();
    };

    const handleAcknowledge = async (alertId: number) => {
        try {
            await fetch(`http://localhost:8000/api/alerts/${alertId}/acknowledge`, {
                method: 'POST'
            });
            // Update local state
            setFetchedAlerts(prev => prev.map(a =>
                a.id === alertId ? { ...a, acknowledged: true } : a
            ));
        } catch (err) {
            console.error('Acknowledge error:', err);
        }
    };

    return (
        <div className="glass-card p-4 h-full">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-accent-danger" />
                    Alerts
                </h2>
                <div className="flex items-center gap-2">
                    <button
                        onClick={fetchAlerts}
                        className="p-1 hover:bg-dark-600 rounded transition-colors"
                        title="Refresh alerts"
                    >
                        <RefreshCw className={`w-4 h-4 text-gray-400 ${isLoading ? 'animate-spin' : ''}`} />
                    </button>
                    <span className="px-2 py-1 rounded-full bg-accent-danger/20 text-accent-danger text-sm">
                        {allAlerts.filter(a => !a.acknowledged).length} New
                    </span>
                </div>
            </div>

            {/* Alert List */}
            <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
                {isLoading && allAlerts.length === 0 ? (
                    <div className="text-center text-gray-400 py-8">
                        <RefreshCw className="w-8 h-8 mx-auto mb-3 animate-spin opacity-50" />
                        <p>Loading alerts...</p>
                    </div>
                ) : error && allAlerts.length === 0 ? (
                    <div className="text-center text-gray-400 py-8">
                        <AlertCircle className="w-12 h-12 mx-auto mb-3 text-accent-danger opacity-50" />
                        <p>{error}</p>
                        <button
                            onClick={fetchAlerts}
                            className="mt-2 text-accent-primary hover:underline text-sm"
                        >
                            Try again
                        </button>
                    </div>
                ) : allAlerts.length === 0 ? (
                    <div className="text-center text-gray-400 py-8">
                        <AlertCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p>No alerts yet</p>
                        <p className="text-sm mt-1">Alerts will appear here in real-time</p>
                    </div>
                ) : (
                    allAlerts.map((alert) => (
                        <div
                            key={alert.id}
                            className={`p-3 rounded-lg ${getSeverityClass(alert.severity)} transition-all hover:scale-[1.02]`}
                        >
                            <div className="flex items-start gap-3">
                                {getSeverityIcon(alert.severity)}
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between">
                                        <span className="font-medium text-sm capitalize">
                                            {alert.alert_type.replace('_', ' ')}
                                        </span>
                                        <span className="text-xs text-gray-400">
                                            {formatTime(alert.timestamp)}
                                        </span>
                                    </div>
                                    <p className="text-sm text-gray-300 mt-1">
                                        {alert.message}
                                    </p>
                                    <div className="flex items-center gap-2 mt-2">
                                        <span className="text-xs px-2 py-0.5 rounded bg-dark-600">
                                            {(alert.confidence * 100).toFixed(0)}% confidence
                                        </span>
                                        {alert.acknowledged ? (
                                            <span className="text-xs text-accent-success flex items-center gap-1">
                                                <Check className="w-3 h-3" /> Acknowledged
                                            </span>
                                        ) : (
                                            <button
                                                onClick={() => handleAcknowledge(alert.id)}
                                                className="text-xs text-accent-primary hover:underline"
                                            >
                                                Acknowledge
                                            </button>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>

            {allAlerts.length > 0 && (
                <button className="w-full mt-4 py-2 text-sm text-accent-primary hover:text-white transition-colors">
                    View All Alerts →
                </button>
            )}
        </div>
    );
}
