import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const useAppStore = create(
  persist(
    (set, get) => ({
      // Theme
      darkMode: window.matchMedia('(prefers-color-scheme: dark)').matches,
      toggleDarkMode: () => set((state) => ({ darkMode: !state.darkMode })),

      // API Keys
      apiKeys: {
        openai: '',
        anthropic: '',
      },
      setApiKeys: (keys) => set({ apiKeys: keys }),

      // Jobs
      jobs: [],
      currentJob: null,
      setJobs: (jobs) => set({ jobs }),
      addJob: (job) => set((state) => ({ jobs: [job, ...state.jobs] })),
      updateJob: (jobId, updates) => set((state) => ({
        jobs: state.jobs.map(job =>
          job.id === jobId ? { ...job, ...updates } : job
        ),
        currentJob: state.currentJob?.id === jobId
          ? { ...state.currentJob, ...updates }
          : state.currentJob
      })),
      setCurrentJob: (job) => set({ currentJob: job }),
      removeJob: (jobId) => set((state) => ({
        jobs: state.jobs.filter(job => job.id !== jobId),
        currentJob: state.currentJob?.id === jobId ? null : state.currentJob
      })),

      // Processing Parameters (preserving all from original)
      parameters: {
        white_level: 201,
        color_tolerance: 100,
        max_blob_area: 2500,
        subtitle_area_height: 0.15,
        crop_sides: 0.20,
        change_threshold: 0.7,
        keyframe_width: 704,
        save_processed_video: true,
      },
      setParameters: (params) => set({ parameters: { ...get().parameters, ...params } }),

      // AI Model
      selectedModel: 'claude-sonnet-4-5-20250929',
      setSelectedModel: (model) => set({ selectedModel: model }),

      // Translation
      translationEnabled: false,
      targetLanguage: null,
      setTranslationEnabled: (enabled) => set({ translationEnabled: enabled }),
      setTargetLanguage: (language) => set({ targetLanguage: language }),

      // UI State
      sidebarOpen: true,
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

      // WebSocket
      wsConnected: false,
      setWsConnected: (connected) => set({ wsConnected: connected }),

      // Notifications
      notifications: [],
      addNotification: (notification) => set((state) => ({
        notifications: [...state.notifications, { id: Date.now(), ...notification }]
      })),
      removeNotification: (id) => set((state) => ({
        notifications: state.notifications.filter(n => n.id !== id)
      })),
    }),
    {
      name: 'captionfuck-storage',
      partialize: (state) => ({
        darkMode: state.darkMode,
        apiKeys: state.apiKeys,
        parameters: state.parameters,
        selectedModel: state.selectedModel,
        sidebarOpen: state.sidebarOpen,
        targetLanguage: state.targetLanguage,
      }),
    }
  )
);