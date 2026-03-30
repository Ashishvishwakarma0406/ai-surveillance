'use client';

import { useState, useEffect } from 'react';
import {
    AlertTriangle, AlertCircle, Info, Check, RefreshCw,
    Swords, Car, Users, Flame, Shield
} from 'lucide-react';

interface Alert {
    id: number;
    alert_type: string;
    severity: string;
    message: string;
    confidence: number;
    category?: string;
    timestamp: string;
    acknowledged: boolean;
    metadata?: Record<string, any>;
}

type CategoryFilter = 'all' | 'violence' | 'traffic';

interface AlertsPanelProps {
    alerts?: Alert[];
    jobIdFilter?: string | null;
    fullPage?: boolean;
}

export default function AlertsPanel({ alerts: wsAlerts = [], jobIdFilter, fullPage = false }: AlertsPanelProps) {
    const [fetchedAlerts, setFetchedAlerts] = useState<Alert[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [activeCategory, setActiveCategory] = useState<CategoryFilter>('all');

    const fetchAlerts = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const endpoint = fullPage 
                ? 'http://localhost:8000/api/alerts?limit=500' 
                : 'http://localhost:8000/api/alerts?limit=50';
            const res = await fetch(endpoint);
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

    // Merge WebSocket alerts with fetched alerts
    let allAlerts = [...wsAlerts, ...fetchedAlerts.filter(
        fetched => !wsAlerts.some(ws => ws.id === fetched.id)
    )].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

    // Filter by active job ID if handling a specific video session
    if (jobIdFilter) {
        allAlerts = allAlerts.filter(a => a.metadata?.job_id === jobIdFilter);
    }

    // Filter by active category
    const filteredAlerts = activeCategory === 'all'
        ? allAlerts
        : allAlerts.filter(a => a.category === activeCategory);

    // Category counts
    const violenceCount = allAlerts.filter(a => a.category === 'violence' && !a.acknowledged).length;
    const trafficCount = allAlerts.filter(a => a.category === 'traffic' && !a.acknowledged).length;
    const totalNew = allAlerts.filter(a => !a.acknowledged).length;

    const getAlertIcon = (alert: Alert) => {
        switch (alert.alert_type) {
            case 'weapon':
                return <Swords className="w-5 h-5 text-red-400" />;
            case 'violence':
                return <Flame className="w-5 h-5 text-red-400" />;
            case 'accident':
                return <Car className="w-5 h-5 text-amber-400" />;
            case 'crowd':
                return <Users className="w-5 h-5 text-blue-400" />;
            case 'intrusion':
                return <Shield className="w-5 h-5 text-red-400" />;
            default:
                return <AlertCircle className="w-5 h-5 text-gray-400" />;
        }
    };

    const getAlertLabel = (alert: Alert) => {
        switch (alert.alert_type) {
            case 'weapon': return 'Danger';
            case 'violence': return 'Violence';
            case 'accident': return 'Collision';
            case 'crowd': return 'Crowd';
            case 'intrusion': return 'Intrusion';
            default: return alert.alert_type;
        }
    };

    const getCardClass = (alert: Alert) => {
        if (alert.category === 'traffic') return 'category-traffic';
        if (alert.category === 'violence') return 'category-violence';
        // Fallback based on severity for older alerts without category
        switch (alert.severity) {
            case 'critical': return 'severity-critical';
            case 'warning': return 'severity-warning';
            default: return 'severity-info';
        }
    };

    const formatTime = (timestamp: string) => {
        return new Date(timestamp).toLocaleTimeString();
    };

    const handleAcknowledge = async (alertId: number) => {
        try {
            await fetch(`http://localhost:8000/api/alerts/${alertId}/acknowledge`, {
                method: 'PATCH'
            });
            setFetchedAlerts(prev => prev.map(a =>
                a.id === alertId ? { ...a, acknowledged: true } : a
            ));
        } catch (err) {
            console.error('Acknowledge error:', err);
        }
    };

    // Render collision signal badges
    const renderSignals = (alert: Alert) => {
        const signals = alert.metadata?.signals;
        if (!signals || alert.alert_type !== 'accident') return null;

        const signalLabels: Record<string, string> = {
            sudden_velocity_drop: 'Velocity Drop',
            proximity_convergence: 'Proximity',
            bbox_overlap: 'Overlap',
            trajectory_disruption: 'Trajectory',
            post_impact_static: 'Post-Impact',
            track_vanished: 'Vanished',
        };

        return (
            <div className="flex flex-wrap mt-1.5">
                {Object.entries(signals).map(([key, active]) => (
                    <span
                        key={key}
                        className={`signal-badge ${active ? 'active' : ''}`}
                    >
                        {signalLabels[key] || key}
                    </span>
                ))}
            </div>
        );
    };

    return (
        <div className="glass-card p-4 h-full">
            <div className="flex items-center justify-between mb-3">
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
                        {totalNew} New
                    </span>
                </div>
            </div>

            {/* Category Filter Tabs */}
            <div className="flex gap-1.5 mb-4">
                <button
                    onClick={() => setActiveCategory('all')}
                    className={`category-tab ${activeCategory === 'all' ? 'active' : ''}`}
                >
                    All
                    {totalNew > 0 && (
                        <span className="ml-1.5 text-xs opacity-60">{totalNew}</span>
                    )}
                </button>
                <button
                    onClick={() => setActiveCategory('violence')}
                    className={`category-tab ${activeCategory === 'violence' ? 'active-violence' : ''}`}
                >
                    <span className="flex items-center gap-1">
                        <Swords className="w-3.5 h-3.5" />
                        Danger
                        {violenceCount > 0 && (
                            <span className="ml-1 px-1.5 py-0.5 rounded-full bg-red-500/20 text-red-400 text-xs">
                                {violenceCount}
                            </span>
                        )}
                    </span>
                </button>
                <button
                    onClick={() => setActiveCategory('traffic')}
                    className={`category-tab ${activeCategory === 'traffic' ? 'active-traffic' : ''}`}
                >
                    <span className="flex items-center gap-1">
                        <Car className="w-3.5 h-3.5" />
                        Crashes
                        {trafficCount > 0 && (
                            <span className="ml-1 px-1.5 py-0.5 rounded-full bg-amber-500/20 text-amber-400 text-xs">
                                {trafficCount}
                            </span>
                        )}
                    </span>
                </button>
            </div>

            {/* Alert List */}
            <div className={`space-y-3 pr-2 overflow-y-auto ${fullPage ? 'h-[75vh]' : 'max-h-[500px]'}`}>
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
                ) : filteredAlerts.length === 0 ? (
                    <div className="text-center text-gray-400 py-8">
                        <AlertCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p>
                            {activeCategory === 'all'
                                ? 'No alerts yet'
                                : `No ${activeCategory === 'violence' ? 'danger' : 'crash'} alerts`
                            }
                        </p>
                        <p className="text-sm mt-1">Alerts will appear here in real-time</p>
                    </div>
                ) : (
                    filteredAlerts.map((alert) => (
                        <div
                            key={alert.id}
                            className={`p-3 rounded-lg ${getCardClass(alert)} transition-all hover:brightness-110`}
                        >
                            <div className="flex items-start gap-3">
                                {getAlertIcon(alert)}
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between">
                                        <span className="font-medium text-sm uppercase tracking-wide">
                                            {getAlertLabel(alert)}
                                        </span>
                                        <span className="text-xs text-gray-400">
                                            {formatTime(alert.timestamp)}
                                        </span>
                                    </div>
                                    <p className="text-sm text-gray-300 mt-1">
                                        {alert.message}
                                    </p>
                                    {/* Collision signal badges */}
                                    {renderSignals(alert)}
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
        </div>
    );
}
