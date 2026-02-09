import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Cameras
export const listCameras = () => api.get('/api/cameras');
export const addCamera = (data: any) => api.post('/api/cameras', data);
export const deleteCamera = (id: string) => api.delete(`/api/cameras/${id}`);
export const startCamera = (id: string) => api.post(`/api/cameras/${id}/start`);
export const stopCamera = (id: string) => api.post(`/api/cameras/${id}/stop`);

// Video Upload
export const uploadVideo = async (file: File, onProgress?: (progress: number) => void) => {
    const formData = new FormData();
    formData.append('file', file);

    return api.post('/api/cameras/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => {
            if (e.total && onProgress) {
                onProgress(Math.round((e.loaded * 100) / e.total));
            }
        },
    });
};

export const getJobStatus = (jobId: string) => api.get(`/api/cameras/jobs/${jobId}`);

// Alerts
export const getAlerts = (params?: any) => api.get('/api/alerts', { params });
export const getAlertStats = () => api.get('/api/alerts/stats');
export const acknowledgeAlert = (id: number) => api.patch(`/api/alerts/${id}/acknowledge`);

// Incidents
export const listIncidents = (params?: any) => api.get('/api/incidents', { params });
export const getIncident = (id: string) => api.get(`/api/incidents/${id}`);
export const createIncident = (data: any) => api.post('/api/incidents', data);
export const updateIncident = (id: string, data: any) => api.patch(`/api/incidents/${id}`, data);

// Health
export const healthCheck = () => api.get('/health');

export default api;
