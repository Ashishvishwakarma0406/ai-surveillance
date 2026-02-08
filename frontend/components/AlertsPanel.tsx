'use client';

import { AlertTriangle, AlertCircle, Info, Check } from 'lucide-react';

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
    alerts: Alert[];
}

export default function AlertsPanel({ alerts }: AlertsPanelProps) {
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

    return (
        <div className="glass-card p-4 h-full">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-accent-danger" />
                    Alerts
                </h2>
                <span className="px-2 py-1 rounded-full bg-accent-danger/20 text-accent-danger text-sm">
                    {alerts.filter(a => !a.acknowledged).length} New
                </span>
            </div>

            {/* Alert List */}
            <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
                {alerts.length === 0 ? (
                    <div className="text-center text-gray-400 py-8">
                        <AlertCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p>No alerts yet</p>
                        <p className="text-sm mt-1">Alerts will appear here in real-time</p>
                    </div>
                ) : (
                    alerts.map((alert) => (
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
                                    <p className="text-sm text-gray-300 mt-1 truncate">
                                        {alert.message}
                                    </p>
                                    <div className="flex items-center gap-2 mt-2">
                                        <span className="text-xs px-2 py-0.5 rounded bg-dark-600">
                                            {(alert.confidence * 100).toFixed(0)}% confidence
                                        </span>
                                        {alert.acknowledged && (
                                            <span className="text-xs text-accent-success flex items-center gap-1">
                                                <Check className="w-3 h-3" /> Acknowledged
                                            </span>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>

            {alerts.length > 0 && (
                <button className="w-full mt-4 py-2 text-sm text-accent-primary hover:text-white transition-colors">
                    View All Alerts →
                </button>
            )}
        </div>
    );
}
