import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Chip,
  IconButton,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  Tooltip,
} from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import {
  Download,
  Delete,
  Refresh,
  CheckCircle,
  Error,
  HourglassEmpty,
  PlayCircleOutline,
  Language,
} from '@mui/icons-material';
import { useAppStore } from '../store/useAppStore';
import { api } from '../api/apiClient';
import toast from 'react-hot-toast';
import { format } from 'date-fns';

function Jobs() {
  const jobs = useAppStore((state) => state.jobs);
  const setJobs = useAppStore((state) => state.setJobs);
  const updateJob = useAppStore((state) => state.updateJob);
  const removeJob = useAppStore((state) => state.removeJob);
  const [loading, setLoading] = useState(false);
  const [deleteDialog, setDeleteDialog] = useState(null);
  const [detailsDialog, setDetailsDialog] = useState(null);

  useEffect(() => {
    fetchJobs();
  }, []);

  const fetchJobs = async () => {
    setLoading(true);
    try {
      const response = await api.getAllJobs();
      setJobs(response.data.jobs);
    } catch (error) {
      console.error('Error fetching jobs:', error);
      toast.error('Failed to load jobs');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (jobId, fileType) => {
    try {
      toast.loading('Downloading...', { id: 'download' });
      const response = await api.downloadFile(jobId, fileType);

      // Create blob and download
      const blob = new Blob([response.data]);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${jobId}_${fileType}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      toast.success('Downloaded!', { id: 'download' });
    } catch (error) {
      console.error('Error downloading file:', error);
      toast.error('Download failed', { id: 'download' });
    }
  };

  const handleDelete = async (jobId) => {
    try {
      await api.deleteJob(jobId);
      removeJob(jobId);
      toast.success('Job deleted');
      setDeleteDialog(null);
    } catch (error) {
      console.error('Error deleting job:', error);
      toast.error('Failed to delete job');
    }
  };

  const handleRetry = async (jobId) => {
    try {
      await api.retryJob(jobId);
      updateJob(jobId, { status: 'pending', progress: 0, error: null });
      toast.success('Job restarted');
    } catch (error) {
      console.error('Error retrying job:', error);
      toast.error('Failed to retry job');
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle sx={{ color: 'success.main' }} />;
      case 'failed':
        return <Error sx={{ color: 'error.main' }} />;
      case 'processing':
      case 'extracting':
      case 'translating':
        return <HourglassEmpty sx={{ color: 'primary.main' }} />;
      default:
        return <PlayCircleOutline sx={{ color: 'text.secondary' }} />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'processing':
      case 'extracting':
      case 'translating':
        return 'primary';
      case 'uploading':
        return 'warning';
      default:
        return 'default';
    }
  };

  const columns = [
    {
      field: 'video_name',
      headerName: 'Video',
      flex: 1,
      minWidth: 200,
      renderCell: (params) => (
        <Box>
          <Typography variant="body2" fontWeight={500}>
            {params.value}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {params.row.id}
          </Typography>
        </Box>
      ),
    },
    {
      field: 'status',
      headerName: 'Status',
      width: 140,
      renderCell: (params) => (
        <Box display="flex" alignItems="center" gap={1}>
          {getStatusIcon(params.value)}
          <Chip
            label={params.value}
            color={getStatusColor(params.value)}
            size="small"
          />
        </Box>
      ),
    },
    {
      field: 'progress',
      headerName: 'Progress',
      width: 150,
      renderCell: (params) => (
        <Box width="100%">
          <LinearProgress
            variant="determinate"
            value={params.value}
            sx={{ mb: 0.5 }}
          />
          <Typography variant="caption">{params.value}%</Typography>
        </Box>
      ),
    },
    {
      field: 'ai_model',
      headerName: 'Model',
      width: 150,
      renderCell: (params) => (
        <Tooltip title={params.value}>
          <Chip
            label={params.value?.includes('claude') ? 'Claude' : 'GPT'}
            size="small"
            variant="outlined"
          />
        </Tooltip>
      ),
    },
    {
      field: 'translation_language',
      headerName: 'Translation',
      width: 120,
      renderCell: (params) => (
        params.value ? (
          <Chip
            icon={<Language />}
            label={params.value}
            size="small"
            color="secondary"
          />
        ) : (
          <Typography variant="caption" color="text.secondary">
            None
          </Typography>
        )
      ),
    },
    {
      field: 'created_at',
      headerName: 'Created',
      width: 140,
      renderCell: (params) => (
        <Typography variant="caption">
          {format(new Date(params.value), 'MMM dd, HH:mm')}
        </Typography>
      ),
    },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 180,
      sortable: false,
      renderCell: (params) => (
        <Box display="flex" gap={0.5}>
          {params.row.status === 'completed' && (
            <>
              <Tooltip title="Download SRT">
                <IconButton
                  size="small"
                  onClick={() => handleDownload(params.row.id, 'srt')}
                >
                  <Download fontSize="small" />
                </IconButton>
              </Tooltip>
              {params.row.translation_language && (
                <Tooltip title="Download Translated SRT">
                  <IconButton
                    size="small"
                    color="secondary"
                    onClick={() => handleDownload(params.row.id, 'translated_srt')}
                  >
                    <Download fontSize="small" />
                  </IconButton>
                </Tooltip>
              )}
            </>
          )}
          {params.row.status === 'failed' && (
            <Tooltip title="Retry">
              <IconButton
                size="small"
                color="warning"
                onClick={() => handleRetry(params.row.id)}
              >
                <Refresh fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
          <Tooltip title="Delete">
            <IconButton
              size="small"
              color="error"
              onClick={() => setDeleteDialog(params.row)}
            >
              <Delete fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      ),
    },
  ];

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4" fontWeight={700}>
            Job History
          </Typography>
          <Typography variant="body2" color="text.secondary">
            View and manage all subtitle extraction jobs
          </Typography>
        </Box>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={fetchJobs}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>

      {/* Stats Cards */}
      <Grid container spacing={2} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h4" color="primary" fontWeight={700}>
                {jobs.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Jobs
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h4" color="success.main" fontWeight={700}>
                {jobs.filter((j) => j.status === 'completed').length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Completed
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h4" color="primary.main" fontWeight={700}>
                {jobs.filter((j) => j.status === 'processing' || j.status === 'extracting').length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Processing
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h4" color="error.main" fontWeight={700}>
                {jobs.filter((j) => j.status === 'failed').length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Failed
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Jobs Table */}
      <Card>
        <CardContent>
          <Box height={600}>
            <DataGrid
              rows={jobs}
              columns={columns}
              pageSize={10}
              rowsPerPageOptions={[10, 25, 50]}
              checkboxSelection={false}
              disableSelectionOnClick
              loading={loading}
              getRowId={(row) => row.id}
              sx={{
                '& .MuiDataGrid-cell:focus': {
                  outline: 'none',
                },
              }}
            />
          </Box>
        </CardContent>
      </Card>

      {/* Delete Confirmation Dialog */}
      <Dialog open={Boolean(deleteDialog)} onClose={() => setDeleteDialog(null)}>
        <DialogTitle>Delete Job?</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this job? This will also delete all output files.
          </Typography>
          {deleteDialog && (
            <Box mt={2}>
              <Typography variant="body2" color="text.secondary">
                Video: {deleteDialog.video_name}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Job ID: {deleteDialog.id}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog(null)}>Cancel</Button>
          <Button
            onClick={() => handleDelete(deleteDialog.id)}
            color="error"
            variant="contained"
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default Jobs;