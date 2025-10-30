@echo off
echo ========================================
echo   CaptionFuck - Railway Deployment
echo ========================================
echo.

cd /d "%~dp0"

echo [1/6] Checking Git...
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found!
    echo Please install Git from: https://git-scm.com/download/win
    pause
    exit /b 1
)
echo [OK] Git found

echo.
echo [2/6] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found!
    echo Please install from: https://nodejs.org/
    pause
    exit /b 1
)
echo [OK] Node.js found

echo.
echo [3/6] Testing frontend build...
cd web_app
if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
)
echo Building frontend...
call npm run build
if errorlevel 1 (
    echo [ERROR] Frontend build failed!
    pause
    exit /b 1
)
cd ..
echo [OK] Frontend builds successfully

echo.
echo [4/6] Initializing Git repository...
if not exist ".git" (
    git init
    echo [OK] Git repository initialized
) else (
    echo [OK] Git repository already exists
)

echo.
echo [5/6] Adding files to Git...
git add .
git commit -m "Ready for Railway deployment" 2>nul
if errorlevel 1 (
    echo [OK] No new changes to commit
) else (
    echo [OK] Files committed
)

echo.
echo [6/6] Deployment package ready!
echo.
echo ========================================
echo   DEPLOYMENT READY!
echo ========================================
echo.
echo Next steps:
echo.
echo 1. Push to GitHub:
echo    git remote add origin https://github.com/yourusername/captionfuck.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo 2. Deploy on Railway:
echo    - Go to railway.app
echo    - New Project ^> Deploy from GitHub
echo    - Select your repo
echo    - Add environment variables:
echo      * OPENAI_API_KEY
echo      * ANTHROPIC_API_KEY
echo    - Deploy!
echo.
echo OR use Railway CLI:
echo    npm install -g @railway/cli
echo    railway login
echo    railway init
echo    railway up
echo.
echo Your app will be live at: https://your-app.up.railway.app
echo.
echo Read DEPLOY_TO_RAILWAY.md for detailed instructions.
echo.
pause