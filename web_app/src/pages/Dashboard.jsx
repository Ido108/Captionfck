import React, { useState, useCallback, useEffect } from 'react';
import {
  Box, Card, CardContent, Typography, Button, LinearProgress, Select, MenuItem,
  FormControl, InputLabel, Accordion, AccordionSummary, AccordionDetails, Slider,
  Grid, Chip, Alert, CircularProgress, TextField, InputAdornment, IconButton,
  Paper, Divider, Stepper, Step, StepLabel, CardActionArea, Fade, Grow,
} from '@mui/material';
import {
  ExpandMore, CloudUpload, PlayArrow, Language, Tune, Key, Visibility, VisibilityOff,
  SmartToy, CheckCircle, Download, VideoFile, Subtitles, Folder, Article, Translate,
  Error as ErrorIcon, Celebration,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { useAppStore } from '../store/useAppStore';
import { api } from '../api/apiClient';
import toast from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';
import ReactPlayer from 'react-player';

const MODELS = [
  { value: 'claude-sonnet-4-5-20250929', label: 'Claude 4.5 Sonnet (Newest)', provider: 'Anthropic' },
  { value: 'claude-sonnet-4-20250514', label: 'Claude 4 Sonnet (Fast)', provider: 'Anthropic' },
  { value: 'claude-3-7-sonnet-20250219', label: 'Claude 3.7 Sonnet (Classic)', provider: 'Anthropic' },
  { value: 'gpt-5-chat-latest', label: 'GPT-5 Chat Latest', provider: 'OpenAI' },
  { value: 'gpt-4.1-2025-04-14', label: 'GPT-4.1', provider: 'OpenAI' },
  { value: 'gpt-4.1-mini', label: 'GPT-4.1 Mini', provider: 'OpenAI' },
  { value: 'gpt-5', label: 'GPT-4o', provider: 'OpenAI' },
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

const PROCESSING_STEPS = [
  { id: 'upload', label: 'Upload' },
  { id: 'extract', label: 'Extract' },
  { id: 'process', label: 'Process' },
  { id: 'translate', label: 'Translate' },
  { id: 'complete', label: 'Complete' },
];

function Dashboard() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [inputVideoUrl, setInputVideoUrl] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingState, setProcessingState] = useState('idle');
  const [showOpenaiKey, setShowOpenaiKey] = useState(false);
  const [showAnthropicKey, setShowAnthropicKey] = useState(false);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [statusMessages, setStatusMessages] = useState([]);

  const selectedModel = useAppStore((state) => state.selectedModel);
  const setSelectedModel = useAppStore((state) => state.setSelectedModel);
  const targetLanguage = useAppStore((state) => state.targetLanguage);
  const setTargetLanguage = useAppStore((state) => state.setTargetLanguage);
  const parameters = useAppStore((state) => state.parameters);
  const setParameters = useAppStore((state) => state.setParameters);
  const apiKeys = useAppStore((state) => state.apiKeys);
  const setApiKeys = useAppStore((state) => state.setApiKeys);
  const addJob = useAppStore((state) => state.addJob);
  const jobs = useAppStore((state) => state.jobs);

  const currentJob = currentJobId ? jobs.find(j => j.id === currentJobId) : null;
  const selectedModelInfo = MODELS.find(m => m.value === selectedModel);
  const needsOpenAI = selectedModelInfo?.provider === 'OpenAI';
  const needsAnthropic = selectedModelInfo?.provider === 'Anthropic';

  useEffect(() => {
    if (currentJob) {
      if (currentJob.status === 'completed') {
        setProcessingState('completed');
        toast.success('🎉 Processing complete!');
      } else if (currentJob.status === 'failed') {
        setProcessingState('failed');
      }
    }
  }, [currentJob]);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setInputVideoUrl(url);
      toast.success(`Selected: ${file.name}`);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'] },
    maxFiles: 1,
    disabled: processingState !== 'idle',
  });

  const handleProcessVideo = async () => {
    if (!selectedFile) {
      toast.error('Please select a video file first');
      return;
    }

    if (needsOpenAI && !apiKeys.openai) {
      toast.error('Please enter your OpenAI API key');
      return;
    }
    if (needsAnthropic && !apiKeys.anthropic) {
      toast.error('Please enter your Anthropic API key');
      return;
    }

    try {
      setProcessingState('uploading');
      setStatusMessages([{ text: 'Starting upload...', type: 'info' }]);

      const uploadResponse = await api.uploadVideo(selectedFile, (progress) => {
        setUploadProgress(progress);
      });

      const { job_id } = uploadResponse.data;

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
      setCurrentJobId(job_id);

      setProcessingState('processing');
      setStatusMessages(prev => [...prev, { text: `Video uploaded successfully. Job ID: ${job_id}`, type: 'success' }]);

      await api.startProcessing(job_id, {
        ai_model: selectedModel,
        parameters: parameters,
        translation: {
          enabled: Boolean(targetLanguage),
          target_language: targetLanguage,
        },
        api_keys: apiKeys,
      });

      setStatusMessages(prev => [...prev, { text: 'Processing started...', type: 'info' }]);

    } catch (error) {
      console.error('Error processing video:', error);
      toast.error(error.response?.data?.detail || 'Failed to process video');
      setProcessingState('failed');
      setStatusMessages(prev => [...prev, { text: `Error: ${error.message}`, type: 'error' }]);
    }
  };

  const handleDownload = async (fileType) => {
    try {
      const response = await api.downloadFile(currentJobId, fileType);
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const a = document.createElement('a');
      a.href = url;

      // Get base filename without extension
      const baseFilename = selectedFile.name.replace(/\.[^/.]+$/, '');

      // Generate proper filename based on file type
      const filenameMap = {
        'srt': `${baseFilename}.srt`,
        'vtt': `${baseFilename}.vtt`,
        'translated_srt': `${baseFilename}_${targetLanguage || 'translated'}.srt`,
        'translated_vtt': `${baseFilename}_${targetLanguage || 'translated'}.vtt`,
        'processed_video': `${baseFilename}_processed.mp4`,
        'log': `${baseFilename}_log.txt`,
      };

      a.download = filenameMap[fileType] || `${baseFilename}_${fileType}`;
      a.click();
      window.URL.revokeObjectURL(url);
      toast.success('Downloaded!');
    } catch (error) {
      toast.error('Download failed');
    }
  };

  const resetDashboard = () => {
    setSelectedFile(null);
    setInputVideoUrl(null);
    setProcessingState('idle');
    setCurrentJobId(null);
    setStatusMessages([]);
    setUploadProgress(0);
  };

  const currentProgress = currentJob?.progress || 0;
  const activeStep = Math.floor(currentProgress / 20);

  return (
    <Box>
      <Typography variant="h4" gutterBottom fontWeight={700}>
        Professional Subtitle Extraction
      </Typography>

      <Grid container spacing={3}>

        {/* Upload Section */}
        {processingState === 'idle' && (
          <Grid item xs={12}>
            <Fade in timeout={500}>
              <Card elevation={3}>
                <CardContent>
                  <Box
                    {...getRootProps()}
                    sx={{
                      border: '3px dashed',
                      borderColor: isDragActive ? 'primary.main' : 'divider',
                      borderRadius: 3,
                      p: 6,
                      textAlign: 'center',
                      cursor: 'pointer',
                      bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                      background: isDragActive
                        ? 'linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%)'
                        : 'transparent',
                      '&:hover': {
                        borderColor: 'primary.main',
                        bgcolor: 'action.hover',
                        transform: 'scale(1.01)',
                      },
                    }}
                  >
                    <input {...getInputProps()} />
                    <CloudUpload sx={{ fontSize: 80, color: 'primary.main', mb: 2 }} />
                    <Typography variant="h5" gutterBottom fontWeight={600}>
                      {isDragActive ? 'Drop video here!' : 'Drag & drop your video'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      or click to browse files
                    </Typography>
                    <Box mt={2}>
                      <Chip label="MP4" size="small" sx={{ mx: 0.5 }} />
                      <Chip label="AVI" size="small" sx={{ mx: 0.5 }} />
                      <Chip label="MOV" size="small" sx={{ mx: 0.5 }} />
                      <Chip label="MKV" size="small" sx={{ mx: 0.5 }} />
                      <Chip label="WebM" size="small" sx={{ mx: 0.5 }} />
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Fade>
          </Grid>
        )}

        {/* Video Input Preview */}
        {selectedFile && inputVideoUrl && (
          <Grid item xs={12}>
            <Grow in timeout={700}>
              <Card elevation={4}>
                <CardContent>
                  <Box display="flex" alignItems="center" gap={1} mb={2}>
                    <VideoFile color="primary" />
                    <Typography variant="h6" fontWeight={600}>
                      Input Video Preview
                    </Typography>
                  </Box>
                  <Box
                    sx={{
                      position: 'relative',
                      paddingTop: '56.25%',
                      bgcolor: 'black',
                      borderRadius: 2,
                      overflow: 'hidden',
                      boxShadow: 3,
                    }}
                  >
                    <ReactPlayer
                      url={inputVideoUrl}
                      controls
                      width="100%"
                      height="100%"
                      style={{ position: 'absolute', top: 0, left: 0 }}
                    />
                  </Box>
                  <Box mt={2} display="flex" gap={1}>
                    <Chip icon={<VideoFile />} label={selectedFile.name} />
                    <Chip label={`${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`} variant="outlined" />
                  </Box>
                </CardContent>
              </Card>
            </Grow>
          </Grid>
        )}

        {/* AI Configuration - Compact Collapsed */}
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
                <Grid item xs={12}>
                  <FormControl fullWidth size="small">
                    <InputLabel>AI Model</InputLabel>
                    <Select
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      label="AI Model"
                      disabled={processingState !== 'idle'}
                    >
                      {MODELS.map((model) => (
                        <MenuItem key={model.value} value={model.value}>
                          <Box display="flex" justifyContent="space-between" width="100%">
                            <Typography variant="body2">{model.label}</Typography>
                            <Chip label={model.provider} size="small" variant="outlined" />
                          </Box>
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth size="small" label="OpenAI API Key"
                    type={showOpenaiKey ? 'text' : 'password'}
                    value={apiKeys.openai}
                    onChange={(e) => setApiKeys({ ...apiKeys, openai: e.target.value })}
                    placeholder="sk-..."
                    disabled={processingState !== 'idle'}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Key fontSize="small" color={apiKeys.openai ? 'success' : 'disabled'} />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton size="small" onClick={() => setShowOpenaiKey(!showOpenaiKey)}>
                            {showOpenaiKey ? <VisibilityOff fontSize="small" /> : <Visibility fontSize="small" />}
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                    helperText={needsOpenAI ? 'Required for selected model' : 'For GPT models'}
                    error={needsOpenAI && !apiKeys.openai}
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth size="small" label="Anthropic API Key"
                    type={showAnthropicKey ? 'text' : 'password'}
                    value={apiKeys.anthropic}
                    onChange={(e) => setApiKeys({ ...apiKeys, anthropic: e.target.value })}
                    placeholder="sk-ant-..."
                    disabled={processingState !== 'idle'}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Key fontSize="small" color={apiKeys.anthropic ? 'success' : 'disabled'} />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton size="small" onClick={() => setShowAnthropicKey(!showAnthropicKey)}>
                            {showAnthropicKey ? <VisibilityOff fontSize="small" /> : <Visibility fontSize="small" />}
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                    helperText={needsAnthropic ? 'Required for selected model' : 'For Claude models'}
                    error={needsAnthropic && !apiKeys.anthropic}
                  />
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
                <Typography fontWeight={600}>Translation (Optional)</Typography>
              </Box>
              <FormControl fullWidth size="small">
                <InputLabel>Target Language</InputLabel>
                <Select
                  value={targetLanguage || ''}
                  onChange={(e) => setTargetLanguage(e.target.value || null)}
                  label="Target Language"
                  disabled={processingState !== 'idle'}
                >
                  {LANGUAGES.map((lang) => (
                    <MenuItem key={lang.code} value={lang.code}>{lang.name}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </CardContent>
          </Card>
        </Grid>

        {/* Processing Progress */}
        {(processingState === 'uploading' || processingState === 'processing') && (
          <Grid item xs={12}>
            <Grow in timeout={500}>
              <Card elevation={4}>
                <CardContent>
                  <Typography variant="h6" gutterBottom fontWeight={600}>
                    {processingState === 'uploading' ? 'Uploading...' : `Processing: ${selectedFile?.name}`}
                  </Typography>

                  <Box mb={3}>
                    <Box display="flex" justifyContent="space-between" mb={1}>
                      <Typography variant="body2" color="text.secondary">
                        {currentJob?.current_step || 'Initializing...'}
                      </Typography>
                      <motion.div animate={{ scale: [1, 1.1, 1] }} transition={{ duration: 1.5, repeat: Infinity }}>
                        <Typography variant="h6" fontWeight={700} color="primary">
                          {processingState === 'uploading' ? uploadProgress : currentProgress}%
                        </Typography>
                      </motion.div>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={processingState === 'uploading' ? uploadProgress : currentProgress}
                      sx={{
                        height: 12,
                        borderRadius: 2,
                        '& .MuiLinearProgress-bar': {
                          background: 'linear-gradient(90deg, #0ea5e9 0%, #10b981 100%)',
                          borderRadius: 2,
                        },
                      }}
                    />
                  </Box>

                  <Stepper activeStep={activeStep} alternativeLabel>
                    {PROCESSING_STEPS.map((step) => (
                      <Step key={step.id}>
                        <StepLabel>{step.label}</StepLabel>
                      </Step>
                    ))}
                  </Stepper>
                </CardContent>
              </Card>
            </Grow>
          </Grid>
        )}

        {/* Status/Output Directory Display */}
        {processingState !== 'idle' && statusMessages.length > 0 && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Folder color="primary" />
                  <Typography fontWeight={600}>Status & Output</Typography>
                </Box>
                <Alert severity="info" sx={{ mb: 2 }}>
                  <Typography variant="caption">
                    Output Directory: <code>{currentJob?.output_files?.process_dir || 'Processing...'}</code>
                  </Typography>
                </Alert>
                <Paper
                  elevation={0}
                  sx={{
                    bgcolor: 'grey.900',
                    color: 'grey.100',
                    p: 2,
                    maxHeight: 200,
                    overflow: 'auto',
                    fontFamily: 'monospace',
                    fontSize: '0.875rem',
                    borderRadius: 1,
                  }}
                >
                  {statusMessages.map((msg, idx) => (
                    <Typography key={idx} variant="caption" component="div" sx={{ mb: 0.5 }}>
                      [{new Date().toLocaleTimeString()}] {msg.text}
                    </Typography>
                  ))}
                </Paper>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Output Videos - Side by Side */}
        {processingState === 'completed' && currentJob?.output_files && (
          <>
            <Grid item xs={12}>
              <Grow in timeout={800}>
                <Alert severity="success" icon={<Celebration />}>
                  <Typography variant="h6" fontWeight={600}>
                    Processing Complete! 🎉
                  </Typography>
                </Alert>
              </Grow>
            </Grid>

            <Grid item xs={12} md={6}>
              <Grow in timeout={1000}>
                <Card elevation={4}>
                  <CardContent>
                    <Box display="flex" alignItems="center" gap={1} mb={2}>
                      <Subtitles color="primary" />
                      <Typography variant="h6" fontWeight={600}>
                        Video with Subtitles
                      </Typography>
                    </Box>
                    <Box
                      sx={{
                        position: 'relative',
                        paddingTop: '56.25%',
                        bgcolor: 'black',
                        borderRadius: 2,
                        overflow: 'hidden',
                      }}
                    >
                      <ReactPlayer
                        url={inputVideoUrl}
                        controls
                        width="100%"
                        height="100%"
                        style={{ position: 'absolute', top: 0, left: 0 }}
                        config={{
                          file: {
                            tracks: currentJob.output_files.vtt ? [
                              { kind: 'subtitles', src: currentJob.output_files.vtt, srcLang: 'en', default: true }
                            ] : []
                          }
                        }}
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grow>
            </Grid>

            <Grid item xs={12} md={6}>
              <Grow in timeout={1200}>
                <Card elevation={4}>
                  <CardContent>
                    <Box display="flex" alignItems="center" gap={1} mb={2}>
                      <VideoFile color="primary" />
                      <Typography variant="h6" fontWeight={600}>
                        Processed B&W Video
                      </Typography>
                    </Box>
                    {currentJob.output_files.processed_video ? (
                      <Box
                        sx={{
                          position: 'relative',
                          paddingTop: '56.25%',
                          bgcolor: 'black',
                          borderRadius: 2,
                          overflow: 'hidden',
                        }}
                      >
                        <video
                          controls
                          style={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            width: '100%',
                            height: '100%',
                          }}
                        >
                          <source src={currentJob.output_files.processed_video} type="video/mp4" />
                        </video>
                      </Box>
                    ) : (
                      <Alert severity="info">Processed video not generated</Alert>
                    )}
                  </CardContent>
                </Card>
              </Grow>
            </Grid>

            {/* Download Cards */}
            <Grid item xs={12}>
              <Typography variant="h5" fontWeight={600} gutterBottom>
                Download Output Files
              </Typography>
              <Grid container spacing={2}>
                {[
                  { key: 'srt', name: 'Subtitles.srt', icon: Subtitles, color: '#0ea5e9' },
                  { key: 'vtt', name: 'Subtitles.vtt', icon: Subtitles, color: '#10b981' },
                  currentJob.translation_language && { key: 'translated_srt', name: 'Translated.srt', icon: Translate, color: '#f59e0b' },
                  currentJob.translation_language && { key: 'translated_vtt', name: 'Translated.vtt', icon: Translate, color: '#f59e0b' },
                  currentJob.output_files.processed_video && { key: 'processed_video', name: 'Processed.mp4', icon: VideoFile, color: '#8b5cf6' },
                  { key: 'log', name: 'Process.log', icon: Article, color: '#64748b' },
                ].filter(Boolean).map((file) => (
                  <Grid item xs={12} sm={6} md={4} lg={2} key={file.key}>
                    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                      <Card
                        elevation={3}
                        sx={{
                          cursor: 'pointer',
                          transition: 'all 0.2s',
                          '&:hover': { boxShadow: 6 },
                        }}
                        onClick={() => handleDownload(file.key)}
                      >
                        <CardContent>
                          <Box display="flex" flexDirection="column" alignItems="center" gap={1.5}>
                            <Box
                              sx={{
                                width: 64,
                                height: 64,
                                borderRadius: 2,
                                bgcolor: file.color,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                color: 'white',
                              }}
                            >
                              <file.icon sx={{ fontSize: 32 }} />
                            </Box>
                            <Typography variant="subtitle2" fontWeight={600} textAlign="center">
                              {file.name}
                            </Typography>
                            <Button
                              variant="contained"
                              size="small"
                              startIcon={<Download />}
                              fullWidth
                            >
                              Download
                            </Button>
                          </Box>
                        </CardContent>
                      </Card>
                    </motion.div>
                  </Grid>
                ))}
              </Grid>
            </Grid>

            {/* Process Another Video Button */}
            <Grid item xs={12}>
              <Button
                variant="contained"
                size="large"
                fullWidth
                startIcon={<CloudUpload />}
                onClick={resetDashboard}
                sx={{ py: 2 }}
              >
                Process Another Video
              </Button>
            </Grid>
          </>
        )}

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
                {[
                  { key: 'white_level', label: 'White Level', min: 180, max: 250, step: 1 },
                  { key: 'color_tolerance', label: 'Color Tolerance', min: 0, max: 150, step: 1 },
                  { key: 'max_blob_area', label: 'Max Blob Area', min: 100, max: 2500, step: 100 },
                  { key: 'subtitle_area_height', label: 'Subtitle Area Height', min: 0.05, max: 0.5, step: 0.01 },
                  { key: 'crop_sides', label: 'Crop Sides', min: 0, max: 0.45, step: 0.01 },
                  { key: 'change_threshold', label: 'Change Threshold %', min: 0.1, max: 10, step: 0.1 },
                  { key: 'keyframe_width', label: 'Keyframe Width (px)', min: 256, max: 1024, step: 64 },
                ].map((param) => (
                  <Grid item xs={12} md={6} key={param.key}>
                    <Typography gutterBottom>
                      {param.label}: {parameters[param.key]}
                    </Typography>
                    <Slider
                      value={parameters[param.key]}
                      onChange={(e, val) => setParameters({ [param.key]: val })}
                      min={param.min}
                      max={param.max}
                      step={param.step}
                      disabled={processingState !== 'idle'}
                    />
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Action Button */}
        {processingState === 'idle' && (
          <Grid item xs={12}>
            <Button
              variant="contained"
              size="large"
              fullWidth
              startIcon={<PlayArrow />}
              onClick={handleProcessVideo}
              disabled={!selectedFile}
              sx={{
                py: 2,
                background: 'linear-gradient(135deg, #0ea5e9 0%, #10b981 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #0284c7 0%, #059669 100%)',
                  boxShadow: 6,
                },
              }}
            >
              Extract Subtitles
            </Button>
          </Grid>
        )}
      </Grid>
    </Box>
  );
}

export default Dashboard;