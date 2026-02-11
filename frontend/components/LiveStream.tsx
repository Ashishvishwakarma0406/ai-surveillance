'use client';

import { useState, useCallback } from 'react';
import { Play, Pause, Maximize2, Camera, RefreshCw, AlertCircle } from 'lucide-react';

export default function LiveStream() {
    const [isStreaming, setIsStreaming] = useState(false);
    const [streamError, setStreamError] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [retryCount, setRetryCount] = useState(0);

    const startStream = useCallback(() => {
        setStreamError(false);
        setIsLoading(true);
        setIsStreaming(true);
    }, []);

    const stopStream = useCallback(() => {
        setIsStreaming(false);
        setIsLoading(false);
        setStreamError(false);
    }, []);

    const toggleStream = () => {
        if (isStreaming) {
            stopStream();
        } else {
            startStream();
        }
    };

    const handleRetry = () => {
        setRetryCount(prev => prev + 1);
        setStreamError(false);
        setIsLoading(true);
    };

    const handleImageLoad = () => {
        setIsLoading(false);
        setStreamError(false);
    };

    const handleImageError = () => {
        setIsLoading(false);
        setStreamError(true);
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
                        disabled={isLoading}
                        className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2 ${isStreaming
                            ? 'bg-accent-danger hover:bg-red-600'
                            : 'bg-accent-success hover:bg-green-600'
                            } ${isLoading ? 'opacity-50 cursor-wait' : ''}`}
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
            <div className="video-container relative">
                {isStreaming && (
                    <img
                        key={retryCount}
                        src={`http://localhost:8000/api/stream/video_feed?source=webcam&t=${retryCount}`}
                        alt="Live Stream"
                        onLoad={handleImageLoad}
                        onError={handleImageError}
                        className={streamError ? 'hidden' : 'w-full h-full object-contain'}
                    />
                )}

                {/* Loading State */}
                {isLoading && isStreaming && !streamError && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400 bg-dark-800/80">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent-primary mb-4"></div>
                        <p>Connecting to camera...</p>
                    </div>
                )}

                {/* Idle State */}
                {!isStreaming && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400">
                        <Camera className="w-16 h-16 mb-4 opacity-50" />
                        <p>Click "Start" to begin streaming</p>
                        <p className="text-sm mt-2 opacity-60">Webcam will be used by default</p>
                    </div>
                )}

                {/* Error State */}
                {streamError && isStreaming && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-dark-800/90">
                        <AlertCircle className="w-12 h-12 text-accent-danger mb-4" />
                        <p className="text-lg font-medium text-accent-danger">Stream Error</p>
                        <p className="text-sm text-gray-400 mt-2 text-center max-w-xs">
                            Could not connect to camera. Make sure:
                        </p>
                        <ul className="text-sm text-gray-500 mt-2 list-disc list-inside">
                            <li>Backend is running on port 8000</li>
                            <li>Camera permissions are granted</li>
                            <li>No other app is using the camera</li>
                        </ul>
                        <button
                            onClick={handleRetry}
                            className="mt-4 px-4 py-2 bg-accent-primary hover:bg-accent-secondary rounded-lg flex items-center gap-2 transition-colors"
                        >
                            <RefreshCw className="w-4 h-4" /> Retry
                        </button>
                    </div>
                )}

                {/* Live Indicator */}
                {isStreaming && !streamError && !isLoading && (
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
