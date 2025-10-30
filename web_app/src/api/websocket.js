import { io } from 'socket.io-client';
import { useAppStore } from '../store/useAppStore';

// Determine WebSocket URL based on environment
const getWebSocketURL = () => {
  if (import.meta.env.VITE_WS_URL) {
    return import.meta.env.VITE_WS_URL;
  }

  // In production, use same host with wss:// or ws://
  if (import.meta.env.PROD) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}`;
  }

  // Development
  return 'ws://localhost:8000';
};

class WebSocketManager {
  constructor() {
    this.socket = null;
    this.clientId = `client_${Math.random().toString(36).substring(7)}`;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 2000;
  }

  connect() {
    if (this.socket?.connected) {
      return;
    }

    try {
      // For WebSocket endpoint in FastAPI
      const WS_BASE_URL = getWebSocketURL();
      const wsUrl = `${WS_BASE_URL}/ws/${this.clientId}`;
      this.socket = new WebSocket(wsUrl);

      this.socket.onopen = () => {
        console.log('WebSocket connected');
        useAppStore.getState().setWsConnected(true);
        this.reconnectAttempts = 0;
      };

      this.socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      this.socket.onclose = () => {
        console.log('WebSocket disconnected');
        useAppStore.getState().setWsConnected(false);
        this.attemptReconnect();
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      this.attemptReconnect();
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);

      setTimeout(() => {
        this.connect();
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('Max reconnection attempts reached');
    }
  }

  handleMessage(data) {
    const { type, job_id, progress, step, status } = data;

    if (type === 'progress') {
      // Update job in store
      useAppStore.getState().updateJob(job_id, {
        progress,
        current_step: step,
        status,
        updated_at: new Date(),
      });

      // Show notification for important events
      if (status === 'completed') {
        useAppStore.getState().addNotification({
          type: 'success',
          title: 'Processing Complete',
          message: `Job ${job_id} finished successfully!`,
        });
      } else if (status === 'failed') {
        useAppStore.getState().addNotification({
          type: 'error',
          title: 'Processing Failed',
          message: `Job ${job_id} encountered an error.`,
        });
      }
    }
  }

  send(data) {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket is not connected');
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }
}

// Singleton instance
const wsManager = new WebSocketManager();

export default wsManager;