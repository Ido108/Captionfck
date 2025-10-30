import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  Grid,
  Alert,
  InputAdornment,
  IconButton,
  Divider,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Visibility,
  VisibilityOff,
  Save,
  CheckCircle,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { api } from '../api/apiClient';
import toast from 'react-hot-toast';

function Settings() {
  const [openaiKey, setOpenaiKey] = useState('');
  const [anthropicKey, setAnthropicKey] = useState('');
  const [showOpenaiKey, setShowOpenaiKey] = useState(false);
  const [showAnthropicKey, setShowAnthropicKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [keyStatus, setKeyStatus] = useState({ openai: false, anthropic: false });

  useEffect(() => {
    fetchKeyStatus();
  }, []);

  const fetchKeyStatus = async () => {
    try {
      const response = await api.getApiKeyStatus();
      setKeyStatus(response.data);
    } catch (error) {
      console.error('Error fetching API key status:', error);
    }
  };

  const handleSave = async () => {
    if (!openaiKey && !anthropicKey) {
      toast.error('Please enter at least one API key');
      return;
    }

    setSaving(true);
    try {
      await api.updateApiKeys({
        openai_key: openaiKey || undefined,
        anthropic_key: anthropicKey || undefined,
      });

      toast.success('API keys saved successfully!');
      setOpenaiKey('');
      setAnthropicKey('');
      fetchKeyStatus();
    } catch (error) {
      console.error('Error saving API keys:', error);
      toast.error('Failed to save API keys');
    } finally {
      setSaving(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom fontWeight={700}>
        Settings
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Configure API keys and application preferences
      </Typography>

      <Grid container spacing={3}>
        {/* API Keys Section */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom fontWeight={600}>
                API Keys
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Enter your API keys to enable subtitle extraction. Your keys are encrypted and stored securely.
              </Typography>

              <Grid container spacing={3}>
                {/* OpenAI API Key */}
                <Grid item xs={12}>
                  <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                    <Typography variant="subtitle1" fontWeight={500}>
                      OpenAI API Key
                    </Typography>
                    {keyStatus.openai && (
                      <Box display="flex" alignItems="center" gap={1}>
                        <CheckCircle color="success" fontSize="small" />
                        <Typography variant="caption" color="success.main">
                          Configured
                        </Typography>
                      </Box>
                    )}
                  </Box>
                  <TextField
                    fullWidth
                    type={showOpenaiKey ? 'text' : 'password'}
                    value={openaiKey}
                    onChange={(e) => setOpenaiKey(e.target.value)}
                    placeholder="sk-..."
                    helperText="Required for GPT models (GPT-4o, GPT-4.1, GPT-5, o4-Mini)"
                    InputProps={{
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            onClick={() => setShowOpenaiKey(!showOpenaiKey)}
                            edge="end"
                          >
                            {showOpenaiKey ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>

                <Grid item xs={12}>
                  <Divider />
                </Grid>

                {/* Anthropic API Key */}
                <Grid item xs={12}>
                  <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                    <Typography variant="subtitle1" fontWeight={500}>
                      Anthropic API Key
                    </Typography>
                    {keyStatus.anthropic && (
                      <Box display="flex" alignItems="center" gap={1}>
                        <CheckCircle color="success" fontSize="small" />
                        <Typography variant="caption" color="success.main">
                          Configured
                        </Typography>
                      </Box>
                    )}
                  </Box>
                  <TextField
                    fullWidth
                    type={showAnthropicKey ? 'text' : 'password'}
                    value={anthropicKey}
                    onChange={(e) => setAnthropicKey(e.target.value)}
                    placeholder="sk-ant-..."
                    helperText="Required for Claude models (Claude 4.5, Claude 4, Claude 3.7)"
                    InputProps={{
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            onClick={() => setShowAnthropicKey(!showAnthropicKey)}
                            edge="end"
                          >
                            {showAnthropicKey ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>

                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    startIcon={<Save />}
                    onClick={handleSave}
                    disabled={saving}
                    size="large"
                  >
                    {saving ? 'Saving...' : 'Save API Keys'}
                  </Button>
                </Grid>
              </Grid>

              <Alert severity="info" sx={{ mt: 3 }}>
                <Typography variant="body2">
                  <strong>How to get API keys:</strong>
                </Typography>
                <Typography variant="body2" mt={1}>
                  • OpenAI: Visit{' '}
                  <a
                    href="https://platform.openai.com/api-keys"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    platform.openai.com/api-keys
                  </a>
                </Typography>
                <Typography variant="body2">
                  • Anthropic: Visit{' '}
                  <a
                    href="https://console.anthropic.com/settings/keys"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    console.anthropic.com/settings/keys
                  </a>
                </Typography>
              </Alert>
            </CardContent>
          </Card>
        </Grid>

        {/* Application Info */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom fontWeight={600}>
                Application Information
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary">
                    Version
                  </Typography>
                  <Typography variant="body1" fontWeight={500}>
                    2.0.0
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary">
                    Backend
                  </Typography>
                  <Typography variant="body1" fontWeight={500}>
                    FastAPI + WebSocket
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary">
                    Supported Models
                  </Typography>
                  <Typography variant="body1" fontWeight={500}>
                    8 AI Models (Claude, GPT)
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary">
                    Translation Languages
                  </Typography>
                  <Typography variant="body1" fontWeight={500}>
                    17 Languages
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Features Overview */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom fontWeight={600}>
                Features
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={4}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <CheckCircle color="success" fontSize="small" />
                    <Typography variant="body2">AI Subtitle Extraction</Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <CheckCircle color="success" fontSize="small" />
                    <Typography variant="body2">17 Language Translation</Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <CheckCircle color="success" fontSize="small" />
                    <Typography variant="body2">Real-time Progress</Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <CheckCircle color="success" fontSize="small" />
                    <Typography variant="body2">Job History Tracking</Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <CheckCircle color="success" fontSize="small" />
                    <Typography variant="body2">SRT & VTT Output</Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <CheckCircle color="success" fontSize="small" />
                    <Typography variant="body2">Dark Mode Support</Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Settings;