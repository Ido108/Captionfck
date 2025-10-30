import React, { useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  Grid,
  Chip,
  Alert,
  CircularProgress,
  TextField,
  InputAdornment,
  IconButton,
} from '@mui/material';
import {
  ExpandMore,
  CloudUpload,
  PlayArrow,
  Language,
  Tune,
  Key,
  Visibility,
  VisibilityOff,
  SmartToy,
  CheckCircle,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { useAppStore } from '../store/useAppStore';
import { api } from '../api/apiClient';
import toast from 'react-hot-toast';

const MODELS = [
  { value: 'claude-sonnet-4-5-20250929', label: 'Claude 4.5 Sonnet (Newest)', provider: 'Anthropic' },
  { value: 'claude-sonnet-4-20250514', label: 'Claude 4 Sonnet (Fast)', provider: 'Anthropic' },
  { value: 'claude-3-7-sonnet-20250219', label: 'Claude 3.7 Sonnet (Classic)', provider: 'Anthropic' },
  { value: 'gpt-5-chat-latest', label: 'GPT-5 Chat Latest', provider: 'OpenAI' },
  { value: 'gpt-4.1-2025-04-14', label: 'GPT-4.1', provider: 'OpenAI' },
  { value: 'gpt-4.1-mini', label: 'GPT-4.1 Mini', provider: 'OpenAI' },
  { value: 'gpt-4o', label: 'GPT-4o', provider: 'OpenAI' },
  { value: 'o4-mini', label: 'o4-Mini', provider: 'OpenAI' },
];

const LANGUAGES = [
  { code: '', name: 'None (No Translation)' },
  { code: 'en', name: 'English' },
  { code: 'he', name: 'Hebrew - עברית' },
  { code: 'es', name: 'Spanish - Español' },
  { code: 'fr', name: 'French - Français' },
  { code: 'de', name: 'German - Deutsch' },
  { code: 'it', name: 'Italian - Italiano' },
  { code: 'pt', name: 'Portuguese - Português' },
  { code: 'ru', name: 'Russian - Русский' },
  { code: 'zh-cn', name: 'Chinese Simplified - 简体中文' },
  { code: 'zh-tw', name: 'Chinese Traditional - 繁體中文' },
  { code: 'ja', name: 'Japanese - 日本語' },
  { code: 'ko', name: 'Korean - 한국어' },
  { code: 'ar', name: 'Arabic - العربية' },
  { code: 'hi', name: 'Hindi - हिन्दी' },
  { code: 'nl', name: 'Dutch - Nederlands' },
  { code: 'tr', name: 'Turkish - Türkçe' },
  { code: 'pl', name: 'Polish - Polski' },
];

function Dashboard() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showOpenaiKey, setShowOpenaiKey] = useState(false);
  const [showAnthropicKey, setShowAnthropicKey] = useState(false);

  // Store
  const selectedModel = useAppStore((state) => state.selectedModel);
  const setSelectedModel = useAppStore((state) => state.setSelectedModel);
  const targetLanguage = useAppStore((state) => state.targetLanguage);
  const setTargetLanguage = useAppStore((state) => state.setTargetLanguage);
  const parameters = useAppStore((state) => state.parameters);
  const setParameters = useAppStore((state) => state.setParameters);
  const apiKeys = useAppStore((state) => state.apiKeys);
  const setApiKeys = useAppStore((state) => state.setApiKeys);
  const addJob = useAppStore((state) => state.addJob);

  // Determine which provider is needed for selected model
  const selectedModelInfo = MODELS.find(m => m.value === selectedModel);
  const needsOpenAI = selectedModelInfo?.provider === 'OpenAI';
  const needsAnthropic = selectedModelInfo?.provider === 'Anthropic';

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0]);
      toast.success(`Selected: ${acceptedFiles[0].name}`);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
    },
    maxFiles: 1,
    disabled: isUploading || isProcessing,
  });

  const handleProcessVideo = async () => {
    if (!selectedFile) {
      toast.error('Please select a video file first');
      return;
    }

    // Validate API keys based on selected model
    if (needsOpenAI && !apiKeys.openai) {
      toast.error('Please enter your OpenAI API key in the configuration section');
      return;
    }
    if (needsAnthropic && !apiKeys.anthropic) {
      toast.error('Please enter your Anthropic API key in the configuration section');
      return;
    }

    try {
      // Upload video
      setIsUploading(true);
      toast.loading('Uploading video...', { id: 'upload' });

      const uploadResponse = await api.uploadVideo(selectedFile, (progress) => {
        setUploadProgress(progress);
      });

      const { job_id } = uploadResponse.data;
      toast.success('Video uploaded successfully!', { id: 'upload' });

      // Add job to store
      const newJob = {
        id: job_id,
        video_name: selectedFile.name,
        status: 'pending',
        progress: 0,
        current_step: 'Waiting to start...',
        created_at: new Date(),
        updated_at: new Date(),
        ai_model: selectedModel,
        translation_language: targetLanguage || null,
        output_files: {},
        parameters: parameters,
      };
      addJob(newJob);

      setIsUploading(false);
      setIsProcessing(true);

      // Start processing
      toast.loading('Starting subtitle extraction...', { id: 'process' });

      await api.startProcessing(job_id, {
        ai_model: selectedModel,
        parameters: parameters,
        translation: {
          enabled: Boolean(targetLanguage),
          target_language: targetLanguage,
        },
        api_keys: apiKeys,
      });

      toast.success('Processing started! Check Job History for progress.', { id: 'process' });

      // Reset form
      setSelectedFile(null);
      setUploadProgress(0);
      setIsProcessing(false);
    } catch (error) {
      console.error('Error processing video:', error);
      toast.error(error.response?.data?.detail || 'Failed to process video', { id: 'upload' });
      setIsUploading(false);
      setIsProcessing(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom fontWeight={700}>
        Upload & Extract Subtitles
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Upload a video with hardcoded subtitles. AI will extract and translate them automatically.
      </Typography>

      <Grid container spacing={3}>
        {/* Upload Section */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box
                {...getRootProps()}
                sx={{
                  border: '2px dashed',
                  borderColor: isDragActive ? 'primary.main' : 'divider',
                  borderRadius: 2,
                  p: 4,
                  textAlign: 'center',
                  cursor: 'pointer',
                  bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                  transition: 'all 0.2s',
                  '&:hover': {
                    borderColor: 'primary.main',
                    bgcolor: 'action.hover',
                  },
                }}
              >
                <input {...getInputProps()} />
                <CloudUpload sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  {isDragActive ? 'Drop video here' : 'Drag & drop video here'}
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  or click to browse
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Supported: MP4, AVI, MOV, MKV, WebM
                </Typography>

                {selectedFile && (
                  <Box mt={2}>
                    <Chip
                      label={selectedFile.name}
                      onDelete={() => setSelectedFile(null)}
                      color="primary"
                      sx={{ mt: 1 }}
                    />
                    <Typography variant="caption" display="block" mt={1}>
                      Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </Typography>
                  </Box>
                )}
              </Box>

              {isUploading && (
                <Box mt={2}>
                  <LinearProgress variant="determinate" value={uploadProgress} />
                  <Typography variant="caption" color="text.secondary" mt={1}>
                    Uploading: {uploadProgress}%
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* API Keys & Model Selection - Compact Collapsed */}
        <Grid item xs={12}>
          <Accordion defaultExpanded={false}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Box display="flex" alignItems="center" gap={1}>
                <SmartToy color="primary" />
                <Typography fontWeight={600}>AI Configuration</Typography>
                {(apiKeys.openai || apiKeys.anthropic) && (
                  <CheckCircle sx={{ fontSize: 16, color: 'success.main', ml: 1 }} />
                )}
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                {/* Model Selection */}
                <Grid item xs={12}>
                  <FormControl fullWidth size="small">
                    <InputLabel>AI Model</InputLabel>
                    <Select
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      label="AI Model"
                      disabled={isUploading || isProcessing}
                    >
                      {MODELS.map((model) => (
                        <MenuItem key={model.value} value={model.value}>
                          <Box display="flex" justifyContent="space-between" width="100%">
                            <Typography variant="body2">{model.label}</Typography>
                            <Chip label={model.provider} size="small" variant="outlined" sx={{ ml: 1 }} />
                          </Box>
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                {/* OpenAI API Key */}
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    size="small"
                    label="OpenAI API Key"
                    type={showOpenaiKey ? 'text' : 'password'}
                    value={apiKeys.openai}
                    onChange={(e) => setApiKeys({ ...apiKeys, openai: e.target.value })}
                    placeholder="sk-..."
                    disabled={isUploading || isProcessing}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Key fontSize="small" color={apiKeys.openai ? 'success' : 'disabled'} />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            size="small"
                            onClick={() => setShowOpenaiKey(!showOpenaiKey)}
                            edge="end"
                          >
                            {showOpenaiKey ? <VisibilityOff fontSize="small" /> : <Visibility fontSize="small" />}
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                    helperText={`For GPT models ${needsOpenAI ? '(Required for selected model)' : ''}`}
                    error={needsOpenAI && !apiKeys.openai}
                  />
                </Grid>

                {/* Anthropic API Key */}
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Anthropic API Key"
                    type={showAnthropicKey ? 'text' : 'password'}
                    value={apiKeys.anthropic}
                    onChange={(e) => setApiKeys({ ...apiKeys, anthropic: e.target.value })}
                    placeholder="sk-ant-..."
                    disabled={isUploading || isProcessing}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Key fontSize="small" color={apiKeys.anthropic ? 'success' : 'disabled'} />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            size="small"
                            onClick={() => setShowAnthropicKey(!showAnthropicKey)}
                            edge="end"
                          >
                            {showAnthropicKey ? <VisibilityOff fontSize="small" /> : <Visibility fontSize="small" />}
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                    helperText={`For Claude models ${needsAnthropic ? '(Required for selected model)' : ''}`}
                    error={needsAnthropic && !apiKeys.anthropic}
                  />
                </Grid>

                <Grid item xs={12}>
                  <Alert severity="info" sx={{ fontSize: '0.875rem' }}>
                    API keys are stored securely in your browser and never sent to our servers. They're only used to call OpenAI/Anthropic directly.
                  </Alert>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Translation */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <Language color="primary" />
                <Typography variant="h6">Translation</Typography>
              </Box>
              <FormControl fullWidth>
                <InputLabel>Target Language (Optional)</InputLabel>
                <Select
                  value={targetLanguage || ''}
                  onChange={(e) => setTargetLanguage(e.target.value || null)}
                  label="Target Language (Optional)"
                  disabled={isUploading || isProcessing}
                >
                  {LANGUAGES.map((lang) => (
                    <MenuItem key={lang.code} value={lang.code}>
                      {lang.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Typography variant="caption" color="text.secondary" mt={1} display="block">
                Leave as "None" to skip translation. Selecting a language will automatically translate the subtitles.
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Advanced Parameters */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Box display="flex" alignItems="center" gap={1}>
                <Tune />
                <Typography>Advanced Parameters</Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>White Level: {parameters.white_level}</Typography>
                  <Slider
                    value={parameters.white_level}
                    onChange={(e, val) => setParameters({ white_level: val })}
                    min={180}
                    max={250}
                    step={1}
                    marks
                    disabled={isUploading || isProcessing}
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>Color Tolerance: {parameters.color_tolerance}</Typography>
                  <Slider
                    value={parameters.color_tolerance}
                    onChange={(e, val) => setParameters({ color_tolerance: val })}
                    min={0}
                    max={150}
                    step={1}
                    disabled={isUploading || isProcessing}
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>Max Blob Area: {parameters.max_blob_area}</Typography>
                  <Slider
                    value={parameters.max_blob_area}
                    onChange={(e, val) => setParameters({ max_blob_area: val })}
                    min={100}
                    max={2500}
                    step={100}
                    disabled={isUploading || isProcessing}
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>Subtitle Area Height: {parameters.subtitle_area_height}</Typography>
                  <Slider
                    value={parameters.subtitle_area_height}
                    onChange={(e, val) => setParameters({ subtitle_area_height: val })}
                    min={0.05}
                    max={0.5}
                    step={0.01}
                    disabled={isUploading || isProcessing}
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>Crop Sides: {parameters.crop_sides}</Typography>
                  <Slider
                    value={parameters.crop_sides}
                    onChange={(e, val) => setParameters({ crop_sides: val })}
                    min={0}
                    max={0.45}
                    step={0.01}
                    disabled={isUploading || isProcessing}
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>Change Threshold: {parameters.change_threshold}%</Typography>
                  <Slider
                    value={parameters.change_threshold}
                    onChange={(e, val) => setParameters({ change_threshold: val })}
                    min={0.1}
                    max={10}
                    step={0.1}
                    disabled={isUploading || isProcessing}
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>Keyframe Width: {parameters.keyframe_width}px</Typography>
                  <Slider
                    value={parameters.keyframe_width}
                    onChange={(e, val) => setParameters({ keyframe_width: val })}
                    min={256}
                    max={1024}
                    step={64}
                    disabled={isUploading || isProcessing}
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Action Button */}
        <Grid item xs={12}>
          <Button
            variant="contained"
            size="large"
            fullWidth
            startIcon={isProcessing ? <CircularProgress size={20} color="inherit" /> : <PlayArrow />}
            onClick={handleProcessVideo}
            disabled={!selectedFile || isUploading || isProcessing}
            sx={{ py: 1.5 }}
          >
            {isProcessing ? 'Processing...' : 'Extract Subtitles'}
          </Button>
          <Typography variant="caption" color="text.secondary" mt={1} display="block">
            Subtitle extraction is always enabled. Translation only happens if you select a language.
          </Typography>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard;