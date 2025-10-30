import axios from 'axios';

// Use relative URLs in production (Railway serves both API and frontend from same domain)
// Use localhost in development
const API_BASE_URL = import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD ? '' : 'http://localhost:8000');

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for video processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // You can add auth tokens here if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with error
      console.error('API Error:', error.response.data);
    } else if (error.request) {
      // Request made but no response
      console.error('Network Error:', error.request);
    } else {
      // Something else happened
      console.error('Error:', error.message);
    }
    return Promise.reject(error);
  }
);

// API methods
export const api = {
  // Models and languages
  getModels: () => apiClient.get('/api/models'),
  getLanguages: () => apiClient.get('/api/languages'),

  // Video upload
  uploadVideo: (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);

    return apiClient.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress(percentCompleted);
        }
      },
    });
  },

  // Processing
  startProcessing: (jobId, config) => {
    return apiClient.post('/api/process', {
      job_id: jobId,
      ai_model: config.ai_model,
      parameters: config.parameters,
      translation: config.translation,
    });
  },

  // Jobs
  getAllJobs: () => apiClient.get('/api/jobs'),
  getJob: (jobId) => apiClient.get(`/api/jobs/${jobId}`),
  deleteJob: (jobId) => apiClient.delete(`/api/jobs/${jobId}`),
  retryJob: (jobId) => apiClient.post(`/api/retry/${jobId}`),

  // Downloads
  downloadFile: (jobId, fileType) => {
    return apiClient.get(`/api/download/${jobId}/${fileType}`, {
      responseType: 'blob',
    });
  },

  // API Keys
  getApiKeyStatus: () => apiClient.get('/api/keys/status'),
  updateApiKeys: (keys) => apiClient.post('/api/keys/update', keys),
};

export default apiClient;