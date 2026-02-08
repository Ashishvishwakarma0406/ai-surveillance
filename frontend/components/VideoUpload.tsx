'use client';

import { useState, useCallback } from 'react';
import { Upload, File, CheckCircle, XCircle, Loader2, X } from 'lucide-react';
import { uploadVideo, getJobStatus } from '@/lib/api';

export default function VideoUpload() {
    const [isDragging, setIsDragging] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [status, setStatus] = useState<'idle' | 'uploading' | 'processing' | 'completed' | 'error'>('idle');
    const [jobId, setJobId] = useState<string | null>(null);
    const [results, setResults] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);
    const [videoUrl, setVideoUrl] = useState<string | null>(null);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile && isValidVideo(droppedFile)) {
            setFile(droppedFile);
        }
    }, []);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile && isValidVideo(selectedFile)) {
            setFile(selectedFile);
        }
    };

    const isValidVideo = (file: File) => {
        const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/webm'];
        return validTypes.includes(file.type);
    };

    const handleUpload = async () => {
        if (!file) return;

        setStatus('uploading');
        setError(null);

        try {
            const response = await uploadVideo(file, setUploadProgress);
            const { job_id } = response.data;
            setJobId(job_id);
            setStatus('processing');

            // Poll for job status
            pollJobStatus(job_id);
        } catch (err: any) {
            setError(err.message || 'Upload failed');
            setStatus('error');
        }
    };

    const pollJobStatus = async (id: string) => {
        try {
            const { data } = await getJobStatus(id);

            if (data.status === 'completed') {
                setStatus('completed');
                setResults(data.results);
                if (data.video_url) {
                    setVideoUrl(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}${data.video_url}`);
                }
            } else if (data.status === 'failed') {
                setStatus('error');
                setError(data.error || 'Processing failed');
            } else {
                // Continue polling
                setTimeout(() => pollJobStatus(id), 2000);
            }
        } catch {
            // Continue polling on error
            setTimeout(() => pollJobStatus(id), 2000);
        }
    };

    const reset = () => {
        setFile(null);
        setStatus('idle');
        setUploadProgress(0);
        setJobId(null);
        setResults(null);
        setError(null);
        setVideoUrl(null);
    };

    return (
        <div className="glass-card p-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5 text-accent-primary" />
                Upload Video
            </h2>

            {(status === 'idle' || status === 'error') && (
                <>
                    {/* Drop Zone */}
                    <div
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${isDragging
                            ? 'border-accent-primary bg-accent-primary/10'
                            : 'border-dark-500 hover:border-dark-400'
                            }`}
                    >
                        <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                        <p className="text-gray-300 mb-2">
                            Drag and drop a video file here
                        </p>
                        <p className="text-sm text-gray-500 mb-4">or</p>
                        <label className="px-6 py-2 bg-accent-primary hover:bg-indigo-600 rounded-lg cursor-pointer transition-colors">
                            Browse Files
                            <input
                                type="file"
                                accept="video/*"
                                onChange={handleFileSelect}
                                className="hidden"
                            />
                        </label>
                        <p className="text-xs text-gray-500 mt-4">
                            Supported: MP4, AVI, MOV, WebM (max 500MB)
                        </p>
                    </div>

                    {/* Selected File */}
                    {/* Selected File */}
                    {file && (status === 'idle' || status === 'error') && (
                        <div className="mt-4 p-4 rounded-lg bg-dark-700 flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <File className="w-8 h-8 text-accent-primary" />
                                <div>
                                    <p className="font-medium">{file.name}</p>
                                    <p className="text-sm text-gray-400">
                                        {(file.size / (1024 * 1024)).toFixed(2)} MB
                                    </p>
                                    {status === 'error' && (
                                        <p className="text-xs text-red-400 mt-1">Upload failed. Try again.</p>
                                    )}
                                </div>
                            </div>
                            <div className="flex gap-2">
                                <button
                                    onClick={reset}
                                    className="p-2 hover:bg-dark-600 rounded-lg transition-colors"
                                    title="Remove file"
                                >
                                    <X className="w-5 h-5 text-gray-400" />
                                </button>
                                <button
                                    onClick={handleUpload}
                                    className="px-6 py-2 bg-accent-success hover:bg-green-600 rounded-lg transition-colors font-medium"
                                >
                                    {status === 'error' ? 'Retry Upload' : 'Process Video'}
                                </button>
                            </div>
                        </div>
                    )}
                </>
            )}

            {/* Uploading */}
            {status === 'uploading' && (
                <div className="text-center py-8">
                    <Loader2 className="w-12 h-12 mx-auto mb-4 text-accent-primary animate-spin" />
                    <p className="text-lg font-medium">Uploading...</p>
                    <div className="w-full max-w-xs mx-auto mt-4 bg-dark-700 rounded-full h-2">
                        <div
                            className="bg-accent-primary h-2 rounded-full transition-all"
                            style={{ width: `${uploadProgress}%` }}
                        />
                    </div>
                    <p className="text-sm text-gray-400 mt-2">{uploadProgress}%</p>
                </div>
            )}

            {/* Processing */}
            {status === 'processing' && (
                <div className="text-center py-8">
                    <Loader2 className="w-12 h-12 mx-auto mb-4 text-accent-warning animate-spin" />
                    <p className="text-lg font-medium">Processing Video...</p>
                    <p className="text-sm text-gray-400 mt-2">
                        Running AI detection on video frames
                    </p>
                </div>
            )}

            {/* Completed */}
            {status === 'completed' && results && (
                <div className="py-4">
                    <div className="flex items-center gap-3 mb-6">
                        <CheckCircle className="w-8 h-8 text-accent-success" />
                        <div>
                            <p className="text-lg font-medium">Processing Complete</p>
                            <p className="text-sm text-gray-400">Analysis results below</p>
                        </div>
                    </div>

                    {/* Results Summary */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                        <div className="p-4 rounded-lg bg-dark-700">
                            <p className="text-2xl font-bold">{results.summary?.max_persons || 0}</p>
                            <p className="text-sm text-gray-400">Max Persons</p>
                        </div>
                        <div className="p-4 rounded-lg bg-dark-700">
                            <p className="text-2xl font-bold text-accent-danger">
                                {results.summary?.max_weapons || 0}
                            </p>
                            <p className="text-sm text-gray-400">Max Weapons</p>
                        </div>
                        <div className="p-4 rounded-lg bg-dark-700">
                            <p className="text-2xl font-bold text-accent-warning">
                                {results.summary?.alert_count || 0}
                            </p>
                            <p className="text-sm text-gray-400">Alerts Generated</p>
                        </div>
                        <div className="p-4 rounded-lg bg-dark-700">
                            <p className="text-2xl font-bold">
                                {results.video_info?.duration?.toFixed(1) || 0}s
                            </p>
                            <p className="text-sm text-gray-400">Duration</p>
                        </div>
                    </div>

                    {/* Video Player */}
                    {videoUrl && (
                        <div className="mb-6">
                            <h3 className="text-md font-semibold mb-2 flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-accent-success" />
                                Processed Video with Detections
                            </h3>
                            <p className="text-sm text-gray-400 mb-3">
                                This video includes bounding boxes for detected persons and weapons.
                            </p>
                            {videoUrl.match(/\.(mp4|webm|ogg)$/i) ? (
                                <video
                                    src={videoUrl}
                                    controls
                                    className="w-full rounded-lg bg-black"
                                    style={{ maxHeight: '400px' }}
                                >
                                    Your browser does not support the video tag.
                                </video>
                            ) : (
                                <div className="p-6 rounded-lg bg-dark-700 text-center">
                                    <p className="text-gray-300 mb-2">
                                        Video format not supported for browser playback
                                    </p>
                                    <p className="text-sm text-gray-500 mb-4">
                                        AVI, MOV, and MKV files cannot be played in browser. Use MP4 or WebM for preview.
                                    </p>
                                    <a
                                        href={videoUrl}
                                        download
                                        className="inline-block px-4 py-2 bg-accent-primary hover:bg-indigo-600 rounded-lg transition-colors"
                                    >
                                        Download Annotated Video
                                    </a>
                                </div>
                            )}
                        </div>
                    )}

                    <button
                        onClick={reset}
                        className="w-full py-3 bg-accent-primary hover:bg-indigo-600 rounded-lg transition-colors"
                    >
                        Upload Another Video
                    </button>
                </div>
            )}

            {/* Error */}
            {status === 'error' && (
                <div className="text-center py-8">
                    <XCircle className="w-12 h-12 mx-auto mb-4 text-accent-danger" />
                    <p className="text-lg font-medium text-accent-danger">Error</p>
                    <p className="text-sm text-gray-400 mt-2">{error}</p>
                    <button
                        onClick={reset}
                        className="mt-4 px-6 py-2 bg-dark-600 hover:bg-dark-500 rounded-lg transition-colors"
                    >
                        Try Again
                    </button>
                </div>
            )}
        </div>
    );
}
