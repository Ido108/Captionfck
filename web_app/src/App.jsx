import React, { useEffect } from 'react';
import {
  Box,
  Container,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Badge,
  Divider,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import {
  Menu as MenuIcon,
  CloudUpload,
  History,
  Settings,
  Brightness4,
  Brightness7,
  Close,
} from '@mui/icons-material';
import { useAppStore } from './store/useAppStore';
import wsManager from './api/websocket';
import Dashboard from './pages/Dashboard';
import Jobs from './pages/Jobs';
import SettingsPage from './pages/Settings';
import toast from 'react-hot-toast';

function App() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [currentPage, setCurrentPage] = React.useState('dashboard');
  const [mobileDrawerOpen, setMobileDrawerOpen] = React.useState(false);

  // Store hooks
  const darkMode = useAppStore((state) => state.darkMode);
  const toggleDarkMode = useAppStore((state) => state.toggleDarkMode);
  const sidebarOpen = useAppStore((state) => state.sidebarOpen);
  const toggleSidebar = useAppStore((state) => state.toggleSidebar);
  const jobs = useAppStore((state) => state.jobs);
  const wsConnected = useAppStore((state) => state.wsConnected);
  const notifications = useAppStore((state) => state.notifications);
  const removeNotification = useAppStore((state) => state.removeNotification);

  // Connect WebSocket on mount
  useEffect(() => {
    wsManager.connect();

    return () => {
      wsManager.disconnect();
    };
  }, []);

  // Handle notifications
  useEffect(() => {
    notifications.forEach((notification) => {
      const toastFn = notification.type === 'error' ? toast.error : toast.success;
      toastFn(notification.message, {
        id: notification.id,
      });
      removeNotification(notification.id);
    });
  }, [notifications, removeNotification]);

  const processingJobsCount = jobs.filter(
    (job) => job.status === 'processing' || job.status === 'uploading'
  ).length;

  const drawerContent = (
    <Box sx={{ width: 250, pt: 2 }}>
      <Typography variant="h6" sx={{ px: 2, mb: 2, fontWeight: 700, color: 'primary.main' }}>
        CaptionFuck
      </Typography>
      <Divider />
      <List>
        <ListItem disablePadding>
          <ListItemButton
            selected={currentPage === 'dashboard'}
            onClick={() => {
              setCurrentPage('dashboard');
              if (isMobile) setMobileDrawerOpen(false);
            }}
          >
            <ListItemIcon>
              <CloudUpload color={currentPage === 'dashboard' ? 'primary' : 'inherit'} />
            </ListItemIcon>
            <ListItemText primary="Upload & Process" />
          </ListItemButton>
        </ListItem>

        <ListItem disablePadding>
          <ListItemButton
            selected={currentPage === 'jobs'}
            onClick={() => {
              setCurrentPage('jobs');
              if (isMobile) setMobileDrawerOpen(false);
            }}
          >
            <ListItemIcon>
              <Badge badgeContent={processingJobsCount} color="primary">
                <History color={currentPage === 'jobs' ? 'primary' : 'inherit'} />
              </Badge>
            </ListItemIcon>
            <ListItemText primary="Job History" />
          </ListItemButton>
        </ListItem>

        <ListItem disablePadding>
          <ListItemButton
            selected={currentPage === 'settings'}
            onClick={() => {
              setCurrentPage('settings');
              if (isMobile) setMobileDrawerOpen(false);
            }}
          >
            <ListItemIcon>
              <Settings color={currentPage === 'settings' ? 'primary' : 'inherit'} />
            </ListItemIcon>
            <ListItemText primary="Settings" />
          </ListItemButton>
        </ListItem>
      </List>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          zIndex: (theme) => theme.zIndex.drawer + 1,
          bgcolor: 'background.paper',
          color: 'text.primary',
          boxShadow: 1,
        }}
      >
        <Toolbar>
          <IconButton
            edge="start"
            color="inherit"
            aria-label="menu"
            onClick={() => (isMobile ? setMobileDrawerOpen(true) : toggleSidebar())}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>

          <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 600 }}>
            Professional Subtitle Extractor
          </Typography>

          {/* WebSocket Status */}
          <Badge
            color={wsConnected ? 'success' : 'error'}
            variant="dot"
            sx={{ mr: 2 }}
            title={wsConnected ? 'Connected' : 'Disconnected'}
          >
            <Box sx={{ width: 8, height: 8 }} />
          </Badge>

          {/* Processing Jobs Counter */}
          {processingJobsCount > 0 && (
            <Badge badgeContent={processingJobsCount} color="primary" sx={{ mr: 2 }}>
              <Typography variant="body2" sx={{ mr: 1 }}>
                Processing
              </Typography>
            </Badge>
          )}

          {/* Theme Toggle */}
          <IconButton onClick={toggleDarkMode} color="inherit">
            {darkMode ? <Brightness7 /> : <Brightness4 />}
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Sidebar - Desktop */}
      {!isMobile && sidebarOpen && (
        <Drawer
          variant="permanent"
          sx={{
            width: 250,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: 250,
              boxSizing: 'border-box',
              mt: '64px',
              height: 'calc(100% - 64px)',
              bgcolor: 'background.paper',
              borderRight: '1px solid',
              borderColor: 'divider',
            },
          }}
        >
          {drawerContent}
        </Drawer>
      )}

      {/* Sidebar - Mobile */}
      {isMobile && (
        <Drawer
          variant="temporary"
          open={mobileDrawerOpen}
          onClose={() => setMobileDrawerOpen(false)}
          ModalProps={{
            keepMounted: true, // Better mobile performance
          }}
          sx={{
            '& .MuiDrawer-paper': {
              width: 250,
              boxSizing: 'border-box',
            },
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', p: 1 }}>
            <IconButton onClick={() => setMobileDrawerOpen(false)}>
              <Close />
            </IconButton>
          </Box>
          {drawerContent}
        </Drawer>
      )}

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          mt: '64px',
          width: isMobile || !sidebarOpen ? '100%' : 'calc(100% - 250px)',
          transition: 'width 0.2s ease',
        }}
      >
        <Container maxWidth="xl">
          {currentPage === 'dashboard' && <Dashboard />}
          {currentPage === 'jobs' && <Jobs />}
          {currentPage === 'settings' && <SettingsPage />}
        </Container>
      </Box>
    </Box>
  );
}

export default App;