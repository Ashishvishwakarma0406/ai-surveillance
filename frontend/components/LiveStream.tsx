'use client';

import { useState } from 'react';
import { Play, Pause, Maximize2, Camera } from 'lucide-react';

export default function LiveStream() {
    const [isStreaming, setIsStreaming] = useState(false);
    const [streamError, setStreamError] = useState(false);

    const toggleStream = () => {
        setStreamError(false);
        setIsStreaming(!isStreaming);
    };

    return (
        <div className="glass-card p-4">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                    <Camera className="w-5 h-5 text-accent-primary" />
                    Live Stream
                </h2>
                <div className="flex gap-2">
                    <button
                        onClick={toggleStream}
                        className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2 ${isStreaming
                                ? 'bg-accent-danger hover:bg-red-600'
                                : 'bg-accent-success hover:bg-green-600'
                            }`}
                    >
                        {isStreaming ? (
                            <>
                                <Pause className="w-4 h-4" /> Stop
                            </>
                        ) : (
                            <>
                                <Play className="w-4 h-4" /> Start
                            </>
                        )}
                    </button>
                    <button className="p-2 rounded-lg glass-card hover:bg-dark-600 transition-colors">
                        <Maximize2 className="w-5 h-5 text-gray-400" />
                    </button>
                </div>
            </div>

            {/* Video Container */}
            <div className="video-container">
                {isStreaming ? (
                    <img
                        src="http://localhost:8000/api/stream/video_feed?source=webcam"
                        alt="Live Stream"
                        onError={() => setStreamError(true)}
                        className={streamError ? 'hidden' : ''}
                    />
                ) : null}

                {!isStreaming && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400">
                        <Camera className="w-16 h-16 mb-4 opacity-50" />
                        <p>Click "Start" to begin streaming</p>
                    </div>
                )}

                {streamError && isStreaming && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-accent-danger">
                        <p className="text-lg font-medium">Stream Error</p>
                        <p className="text-sm text-gray-400 mt-2">
                            Make sure the backend is running on port 8000
                        </p>
                    </div>
                )}

                {/* Live Indicator */}
                {isStreaming && !streamError && (
                    <div className="absolute top-4 left-4 px-3 py-1 rounded-full bg-accent-danger flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
                        <span className="text-sm font-medium">LIVE</span>
                    </div>
                )}
            </div>

            {/* Stream Info */}
            <div className="mt-4 flex items-center justify-between text-sm text-gray-400">
                <span>Source: Webcam (Default)</span>
                <span>Resolution: 1280x720</span>
            </div>
        </div>
    );
}
