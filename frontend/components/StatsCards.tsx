'use client';

import { Activity, Swords, Car, Camera, Clock } from 'lucide-react';

interface Stats {
    totalAlerts: number;
    activeStreams: number;
    detections: number;
    uptime: string;
    dangerAlerts?: number;
    crashAlerts?: number;
}

interface StatsCardsProps {
    stats: Stats;
}

export default function StatsCards({ stats }: StatsCardsProps) {
    const cards = [
        {
            title: 'Danger Alerts',
            value: stats.dangerAlerts ?? stats.totalAlerts,
            icon: Swords,
            color: 'text-red-400',
            bgColor: 'bg-red-500/10'
        },
        {
            title: 'Crash Alerts',
            value: stats.crashAlerts ?? 0,
            icon: Car,
            color: 'text-amber-400',
            bgColor: 'bg-amber-500/10'
        },
        {
            title: 'Active Streams',
            value: stats.activeStreams,
            icon: Camera,
            color: 'text-accent-success',
            bgColor: 'bg-accent-success/10'
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
