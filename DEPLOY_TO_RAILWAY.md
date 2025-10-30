# 🚂 Deploy CaptionFuck to Railway - Complete Guide

## ✅ What's in This Folder

This `deploy_railway` folder contains **EVERYTHING** needed for Railway deployment:

```
deploy_railway/
├── main.py              ✅ Production entry point (FastAPI)
├── app.py               ✅ Core processing logic (copied from parent)
├── prompt.txt           ✅ AI prompt template
├── requirements.txt     ✅ Python dependencies
├── Procfile             ✅ Railway start command
├── nixpacks.toml        ✅ Build configuration
├── railway.json         ✅ Railway config
├── .gitignore           ✅ Git ignore rules
├── .env.example         ✅ Environment variables template
└── web_app/             ✅ Complete React frontend
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── src/
        └── (all React components)
```

---

## 🚀 DEPLOYMENT STEPS (5 Minutes!)

### Step 1: Create Railway Account
1. Go to https://railway.app
2. Sign up with GitHub (recommended)
3. Verify your email

### Step 2: Initialize Git Repository

Open terminal in this `deploy_railway` folder:

```bash
cd C:\Users\ido\CaptionFuck\deploy_railway

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - CaptionFuck deployment"
```

### Step 3: Deploy to Railway

**Option A: Using Railway CLI (Recommended)**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create new project
railway init

# Deploy
railway up

# Open in browser
railway open
```

**Option B: Using GitHub**

1. Create a new GitHub repository
2. Push this folder:
   ```bash
   git remote add origin https://github.com/yourusername/captionfuck.git
   git branch -M main
   git push -u origin main
   ```
3. Go to Railway dashboard
4. Click "New Project"
5. Choose "Deploy from GitHub repo"
6. Select your repository
7. Railway will auto-detect and build!

### Step 4: Set Environment Variables in Railway

Go to your Railway project → Variables → Add:

```
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
ENVIRONMENT=production
```

### Step 5: Done!

Railway will give you a URL like: `https://your-app.up.railway.app`

Your app is now **LIVE ON THE INTERNET!** 🎉

---

## 🌐 How Railway Deploys This

1. **Detects** Python + Node.js project (via nixpacks.toml)
2. **Installs** Python dependencies from requirements.txt
3. **Installs** Node.js dependencies (web_app/package.json)
4. **Builds** React frontend (`npm run build`)
5. **Serves** Frontend from main.py (FastAPI serves static files)
6. **Starts** Backend with `uvicorn main:app`
7. **Exposes** on Railway's domain

---

## 📝 Important Notes

### API Keys
- **DO NOT** commit API keys to git
- Set them as environment variables in Railway dashboard
- They're loaded from Railway environment automatically

### Port
- Railway sets the `PORT` environment variable automatically
- main.py uses this: `port = int(os.environ.get("PORT", 8000))`

### Static Files
- Frontend is built during deployment
- Served from `web_app/dist/` by FastAPI
- All requests go through main.py

### WebSocket
- Railway supports WebSocket automatically
- No special configuration needed
- WS connections work on same domain

---

## 🔧 Configuration Files Explained

### `main.py`
- Production FastAPI app
- Imports from app.py
- Serves static frontend files
- Handles WebSocket connections
- Uses environment PORT variable

### `requirements.txt`
- All Python dependencies
- Uses production versions
- opencv-python-headless for Railway (no GUI needed)

### `Procfile`
- Tells Railway how to start the app
- Uses uvicorn to run FastAPI
- Binds to Railway's PORT

### `nixpacks.toml`
- Build configuration for Railway
- Installs Python + Node.js
- Builds frontend during deployment
- Sets start command

### `railway.json`
- Railway-specific settings
- Build and deploy commands
- Restart policy

### `.gitignore`
- Excludes node_modules, venv, output files
- Keeps repository clean
- Protects sensitive files

---

## 🎯 Production Checklist

Before deploying, make sure:

- ✅ All files copied to deploy_railway folder
- ✅ .gitignore prevents committing secrets
- ✅ requirements.txt has all dependencies
- ✅ Frontend builds successfully (`cd web_app && npm run build`)
- ✅ Environment variables ready (API keys)
- ✅ Git repository initialized
- ✅ README included for documentation

---

## 🧪 Test Locally First

Before deploying to Railway, test locally:

```bash
cd C:\Users\ido\CaptionFuck\deploy_railway

# Install dependencies
pip install -r requirements.txt
cd web_app && npm install && npm run build && cd ..

# Run locally
python main.py

# Open http://localhost:8000
```

If it works locally, it will work on Railway!

---

## 🚨 Troubleshooting

### Build Fails
- Check Railway build logs
- Verify requirements.txt has all dependencies
- Ensure Node.js version is compatible

### Frontend Not Loading
- Check that `web_app/dist/` exists after build
- Verify Vite config is correct
- Check Railway logs for static file errors

### WebSocket Not Connecting
- Railway supports WebSocket by default
- No configuration needed
- Check browser console for errors

### API Keys Not Working
- Verify they're set in Railway environment variables
- Check spelling (OPENAI_API_KEY, ANTHROPIC_API_KEY)
- Keys should NOT be in code or config.ini

### Upload Fails
- Railway has ephemeral filesystem
- Uploads work but files are deleted on restart
- For production, use S3 or persistent storage

---

## 📊 Railway Free Tier

- ✅ $5 free credit per month
- ✅ Enough for testing and light usage
- ✅ Can upgrade for more resources
- ✅ Automatic HTTPS
- ✅ Custom domains supported

---

## 🎉 Success!

Once deployed, you can:
- Access from anywhere with internet
- Share the URL with others
- Use on mobile devices
- No local setup needed for users

Your professional subtitle extraction tool is now **ONLINE!** 🌍

---

## 📞 Support

Railway Documentation: https://docs.railway.app
Railway Discord: https://discord.gg/railway

For app-specific issues, check the logs in Railway dashboard.

---

**Ready to deploy? Just follow Step 2-5 above!** 🚀