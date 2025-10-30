# ✅ Railway Deployment Checklist

## Pre-Deployment Checklist

### Local Setup
- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] Git installed
- [ ] Railway account created (railway.app)

### Files Ready
- [x] main.py exists
- [x] app.py exists
- [x] prompt.txt exists
- [x] requirements.txt exists
- [x] web_app/ folder exists
- [x] All configuration files present

### Test Locally
- [ ] Run: `pip install -r requirements.txt`
- [ ] Run: `cd web_app && npm install && npm run build && cd ..`
- [ ] Run: `python main.py`
- [ ] Open: http://localhost:8000
- [ ] Verify frontend loads
- [ ] Test video upload
- [ ] Test subtitle extraction

---

## Deployment Checklist

### Git Setup
- [ ] Run: `git init`
- [ ] Run: `git add .`
- [ ] Run: `git commit -m "Initial deployment"`
- [ ] Create GitHub repository
- [ ] Run: `git remote add origin <your-repo-url>`
- [ ] Run: `git push -u origin main`

### Railway Setup
- [ ] Go to railway.app
- [ ] Create new project
- [ ] Deploy from GitHub repo
- [ ] Wait for build to complete

### Environment Variables
Set these in Railway dashboard → Variables:
- [ ] OPENAI_API_KEY = (your OpenAI key)
- [ ] ANTHROPIC_API_KEY = (your Anthropic key)
- [ ] ENVIRONMENT = production

### Verify Deployment
- [ ] Frontend loads at Railway URL
- [ ] API endpoints respond (/api/models, /api/languages)
- [ ] WebSocket connects
- [ ] Can upload a test video
- [ ] Can select AI model
- [ ] Processing works end-to-end
- [ ] Can download SRT files
- [ ] Dark mode works
- [ ] Mobile version works

---

## Post-Deployment Checklist

### Monitoring
- [ ] Check Railway logs for errors
- [ ] Monitor API usage (OpenAI/Anthropic dashboards)
- [ ] Test from different devices
- [ ] Test from different locations

### Optional Enhancements
- [ ] Add custom domain
- [ ] Set up database for job persistence
- [ ] Add S3 for file storage
- [ ] Set up monitoring/analytics
- [ ] Add rate limiting
- [ ] Add authentication (if needed)

---

## ✅ Success Indicators

If these all work, deployment is successful:

1. ✅ https://your-app.up.railway.app loads
2. ✅ Can upload video via drag & drop
3. ✅ Can select from 8 AI models
4. ✅ Can choose from 17 languages
5. ✅ Processing shows real-time progress
6. ✅ Can download SRT/VTT files
7. ✅ Job history displays correctly
8. ✅ Settings page works
9. ✅ Dark mode toggles
10. ✅ Mobile responsive

---

## 🚨 If Something Fails

### Build Fails
1. Check Railway build logs
2. Verify requirements.txt syntax
3. Ensure web_app/package.json is valid
4. Check nixpacks.toml configuration

### App Doesn't Start
1. Check Railway deployment logs
2. Verify environment variables are set
3. Check main.py for errors
4. Verify Procfile is correct

### Frontend Not Loading
1. Check if web_app/dist/ was created during build
2. Verify static file serving in main.py
3. Check Railway logs for 404 errors
4. Rebuild frontend: `npm run build`

### API Not Working
1. Check environment variables (API keys)
2. Verify /api/health endpoint
3. Check Railway logs for Python errors
4. Test API endpoints individually

---

## 🎓 Quick Reference

**Railway Dashboard:**
- Logs: View real-time logs
- Metrics: CPU, Memory, Network usage
- Variables: Environment variables
- Deployments: Deployment history

**URLs:**
- App: https://your-app.up.railway.app
- API Docs: https://your-app.up.railway.app/docs (FastAPI auto-generated)
- Health: https://your-app.up.railway.app/api/health

**Commands:**
```bash
railway login          # Login to Railway
railway init           # Create project
railway up             # Deploy
railway logs           # View logs
railway open           # Open in browser
railway status         # Check status
railway env            # View environment variables
```

---

## 🎉 Deployment Complete!

Once all checkboxes are ✅, your app is **LIVE ON THE INTERNET**!

Share your Railway URL with anyone, anywhere in the world! 🌍

---

**Need help? Read DEPLOY_TO_RAILWAY.md for detailed instructions.**