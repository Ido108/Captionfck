# CaptionFuck - Railway Deployment Package

## 🚀 Super Quick Deploy

### 1. Prerequisites
- Railway account (https://railway.app)
- GitHub account (for easy deployment)
- Your API keys (OpenAI and/or Anthropic)

### 2. Deploy in 2 Minutes

```bash
# In this folder (deploy_railway)
git init
git add .
git commit -m "Deploy CaptionFuck"

# Push to GitHub
git remote add origin https://github.com/yourusername/captionfuck.git
git push -u origin main

# Go to Railway:
# 1. New Project → Deploy from GitHub
# 2. Select your repository
# 3. Add environment variables:
#    - OPENAI_API_KEY=your-key
#    - ANTHROPIC_API_KEY=your-key
# 4. Deploy!
```

### 3. Your App is Live!

Railway gives you a URL like: `https://your-app.up.railway.app`

---

## ✨ What's Included

✅ **Complete Backend** - FastAPI with all features
✅ **Complete Frontend** - React app with Material UI
✅ **8 AI Models** - Claude 4.5, GPT-5, etc.
✅ **17 Languages** - Translation support
✅ **Real-time Updates** - WebSocket support
✅ **Job History** - Track all extractions
✅ **Dark Mode** - Professional UI

---

## 📁 Folder Structure

```
deploy_railway/
├── main.py                 # Production entry point
├── app.py                  # Core processing logic
├── prompt.txt              # AI prompt
├── requirements.txt        # Python deps
├── Procfile                # Railway start
├── nixpacks.toml           # Build config
├── railway.json            # Railway settings
├── .gitignore              # Git ignore
├── .env.example            # Env template
└── web_app/                # React frontend
    ├── package.json
    ├── vite.config.js
    └── src/
```

---

## ⚙️ Environment Variables (Set in Railway)

**Required:**
- `OPENAI_API_KEY` - For GPT models
- `ANTHROPIC_API_KEY` - For Claude models

**Optional:**
- `PORT` - Auto-set by Railway
- `ENVIRONMENT` - Set to "production"

---

## 🔥 Features

### For Users
- Upload videos from any device
- Extract hardcoded subtitles with AI
- Translate to 17 languages
- Download SRT/VTT files
- Track job history
- Works on mobile!

### Technical
- FastAPI backend (async, fast)
- React frontend (modern, responsive)
- WebSocket (real-time progress)
- Material UI (professional design)
- Production-ready
- Scalable architecture

---

## 📖 Documentation

- `DEPLOY_TO_RAILWAY.md` - Detailed deployment guide
- `.env.example` - Environment variables template

---

## 🎯 Quick Commands

```bash
# Test locally before deploying
pip install -r requirements.txt
cd web_app && npm install && npm run build && cd ..
python main.py

# Deploy to Railway
railway login
railway init
railway up

# View logs
railway logs

# Open deployed app
railway open
```

---

## ✅ Deployment Checklist

Before pushing to Railway:

- [ ] Git repository initialized
- [ ] All files committed
- [ ] .gitignore properly configured
- [ ] Frontend builds successfully
- [ ] Backend runs locally
- [ ] Railway project created
- [ ] Environment variables set
- [ ] API keys added to Railway

---

## 🌟 That's It!

Your professional subtitle extraction tool will be **LIVE ON THE INTERNET** with just a few commands!

**Happy deploying! 🚀**