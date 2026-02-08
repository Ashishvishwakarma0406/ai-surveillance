'use client';

import { Activity, AlertTriangle, Camera, Clock } from 'lucide-react';

interface Stats {
    totalAlerts: number;
    activeStreams: number;
    detections: number;
    uptime: string;
}

interface StatsCardsProps {
    stats: Stats;
}

export default function StatsCards({ stats }: StatsCardsProps) {
    const cards = [
        {
            title: 'Total Alerts',
            value: stats.totalAlerts,
            icon: AlertTriangle,
            color: 'text-accent-danger',
            bgColor: 'bg-accent-danger/10'
        },
        {
            title: 'Active Streams',
            value: stats.activeStreams,
            icon: Camera,
            color: 'text-accent-success',
            bgColor: 'bg-accent-success/10'
        },
        {
            title: 'Detections',
            value: stats.detections,
            icon: Activity,
            color: 'text-accent-primary',
            bgColor: 'bg-accent-primary/10'
        },
        {
            title: 'Uptime',
            value: stats.uptime,
            icon: Clock,
            color: 'text-accent-warning',
            bgColor: 'bg-accent-warning/10'
        }
    ];

    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {cards.map((card) => (
                <div key={card.title} className="glass-card p-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-gray-400">{card.title}</p>
                            <p className="text-2xl font-bold mt-1">{card.value}</p>
                        </div>
                        <div className={`p-3 rounded-xl ${card.bgColor}`}>
                            <card.icon className={`w-6 h-6 ${card.color}`} />
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
}
