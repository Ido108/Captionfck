#!/usr/bin/env python3
# Production Entry Point for Railway Deployment

import os
import sys
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import json
import shutil
import uuid
from datetime import datetime
import logging

# Import from app_core.py (no Gradio, only processing functions)
from app_core import (
    extract_subtitle_keyframes,
    generate_srt_from_keyframes,
    translate_srt,
    convert_srt_to_vtt,
    create_process_output_dir,
    setup_logger,
    get_api_keys,
    save_api_keys,
    APP_DIR,
    PROMPT_FILE_NAME,
    MODEL_CLAUDE_45_SONNET,
    MODEL_CLAUDE_4_SONNET,
    MODEL_CLAUDE_37_SONNET,
    MODEL_GPT5_CHAT,
    MODEL_GPT41,
    MODEL_GPT41_MINI,
    MODEL_GPT4O,
    MODEL_O4_MINI
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="CaptionFuck API",
    version="2.0.0",
    description="Professional subtitle extraction and translation service"
)

# CORS - Allow all origins for Railway deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job storage
jobs_db = {}
active_websockets = {}

# Enums
class JobStatus(str):
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    EXTRACTING = "extracting"
    TRANSLATING = "translating"
    COMPLETED = "completed"
    FAILED = "failed"

# Models
class ProcessingParameters(BaseModel):
    white_level: int = 201
    color_tolerance: int = 100
    max_blob_area: int = 2500
    subtitle_area_height: float = 0.15
    crop_sides: float = 0.20
    change_threshold: float = 0.7
    keyframe_width: int = 704
    save_processed_video: bool = True

class TranslationSettings(BaseModel):
    enabled: bool = False
    target_language: Optional[str] = None

class ExtractionRequest(BaseModel):
    job_id: str
    ai_model: str
    parameters: ProcessingParameters = ProcessingParameters()
    translation: TranslationSettings = TranslationSettings()
    api_keys: Dict[str, str] = {}

class ApiKeyUpdate(BaseModel):
    openai_key: Optional[str] = None
    anthropic_key: Optional[str] = None

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket disconnected: {client_id}")

    async def send_json(self, data: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(data)
            except:
                self.disconnect(client_id)

manager = ConnectionManager()

# Helper functions
def create_job_id():
    return f"job_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

async def update_job_progress(job_id: str, progress: int, step: str, status: str = None):
    if job_id in jobs_db:
        jobs_db[job_id]["progress"] = progress
        jobs_db[job_id]["current_step"] = step
        jobs_db[job_id]["updated_at"] = datetime.now()
        if status:
            jobs_db[job_id]["status"] = status

        update_data = {
            "type": "progress",
            "job_id": job_id,
            "progress": progress,
            "step": step,
            "status": status or jobs_db[job_id]["status"]
        }

        for client_id in list(active_websockets.keys()):
            await manager.send_json(update_data, client_id)

# API Endpoints
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}

@app.get("/api/models")
async def get_available_models():
    return {
        "models": [
            {"value": MODEL_CLAUDE_45_SONNET, "label": "Claude 4.5 Sonnet (Newest)", "provider": "Anthropic"},
            {"value": MODEL_CLAUDE_4_SONNET, "label": "Claude 4 Sonnet (Fast)", "provider": "Anthropic"},
            {"value": MODEL_CLAUDE_37_SONNET, "label": "Claude 3.7 Sonnet (Classic)", "provider": "Anthropic"},
            {"value": MODEL_GPT5_CHAT, "label": "GPT-5 Chat Latest", "provider": "OpenAI"},
            {"value": MODEL_GPT41, "label": "GPT-4.1", "provider": "OpenAI"},
            {"value": MODEL_GPT41_MINI, "label": "GPT-4.1 Mini", "provider": "OpenAI"},
            {"value": MODEL_GPT4O, "label": "GPT-4o", "provider": "OpenAI"},
            {"value": MODEL_O4_MINI, "label": "o4-Mini", "provider": "OpenAI"}
        ]
    }

@app.get("/api/languages")
async def get_supported_languages():
    return {
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "he", "name": "Hebrew"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "zh-cn", "name": "Chinese (Simplified)"},
            {"code": "zh-tw", "name": "Chinese (Traditional)"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "ar", "name": "Arabic"},
            {"code": "hi", "name": "Hindi"},
            {"code": "nl", "name": "Dutch"},
            {"code": "tr", "name": "Turkish"},
            {"code": "pl", "name": "Polish"}
        ]
    }

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise HTTPException(status_code=400, detail="Invalid video format")

        job_id = create_job_id()
        upload_dir = os.path.join(BASE_DIR, "uploads", job_id)
        os.makedirs(upload_dir, exist_ok=True)
        video_path = os.path.join(upload_dir, file.filename)

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        job = {
            "id": job_id,
            "status": "pending",
            "video_name": file.filename,
            "video_path": video_path,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "progress": 0,
            "current_step": "Video uploaded successfully",
            "output_files": {},
            "error": None
        }

        jobs_db[job_id] = job
        logger.info(f"Video uploaded: {job_id} - {file.filename}")

        return {"job_id": job_id, "filename": file.filename, "status": "uploaded"}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process")
async def start_processing(request: ExtractionRequest, background_tasks: BackgroundTasks):
    job_id = request.job_id

    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    jobs_db[job_id]["ai_model"] = request.ai_model
    jobs_db[job_id]["parameters"] = request.parameters.dict()
    jobs_db[job_id]["translation_language"] = request.translation.target_language

    background_tasks.add_task(process_video_task, job_id, request.ai_model, request.parameters, request.translation, request.api_keys)

    return {"status": "processing_started", "job_id": job_id}

async def process_video_task(job_id: str, ai_model: str, parameters: ProcessingParameters, translation: TranslationSettings, api_keys_from_request: Dict[str, str]):
    try:
        job = jobs_db[job_id]
        video_path = job["video_path"]

        await update_job_progress(job_id, 10, "Initializing processing...", "processing")

        process_dir, keyframes_dir, composite_dir = create_process_output_dir()
        log_file = os.path.join(process_dir, "process.log")
        logger_inst = setup_logger(log_file)

        await update_job_progress(job_id, 20, "Extracting keyframes...", "extracting")

        individual_keyframes, segments, processed_video_path, composite_paths = extract_subtitle_keyframes(
            src=video_path,
            process_dir=process_dir,
            keyframes_dir=keyframes_dir,
            composite_keyframes_dir=composite_dir,
            logger=logger_inst,
            white_lvl=parameters.white_level,
            tol=parameters.color_tolerance,
            keep_ratio_h=parameters.subtitle_area_height,
            side_ratio=parameters.crop_sides,
            min_blob=parameters.max_blob_area,
            change_percent_threshold=parameters.change_threshold,
            target_frame_width=parameters.keyframe_width,
            save_processed_video=parameters.save_processed_video
        )

        await update_job_progress(job_id, 50, "Generating subtitles with AI...")

        # Use API keys from request (stored in browser)
        prompt_file = os.path.join(BASE_DIR, PROMPT_FILE_NAME)
        srt_file = generate_srt_from_keyframes(
            composite_image_paths=composite_paths,
            time_segments=segments,
            api_keys=api_keys_from_request,
            prompt_file_path=prompt_file,
            process_dir=process_dir,
            logger=logger_inst,
            model=ai_model
        )

        vtt_file = convert_srt_to_vtt(srt_file, logger_inst) if srt_file else None

        translated_srt = None
        translated_vtt = None
        if translation.enabled and translation.target_language:
            await update_job_progress(job_id, 70, f"Translating to {translation.target_language}...", "translating")
            translated_srt, translated_vtt = translate_srt(srt_file, translation.target_language, api_keys_from_request, logger_inst, model=ai_model)

        await update_job_progress(job_id, 90, "Finalizing...")

        output_files = {"srt": srt_file, "vtt": vtt_file, "log": log_file, "process_dir": process_dir}
        if processed_video_path:
            output_files["processed_video"] = processed_video_path
        if translated_srt:
            output_files["translated_srt"] = translated_srt
            output_files["translated_vtt"] = translated_vtt

        jobs_db[job_id]["output_files"] = output_files
        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["progress"] = 100
        jobs_db[job_id]["current_step"] = "Completed!"

        await update_job_progress(job_id, 100, "Completed!", "completed")
        logger.info(f"Job completed: {job_id}")

    except Exception as e:
        logger.error(f"Processing error for job {job_id}: {str(e)}")
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = str(e)
        await update_job_progress(job_id, 0, f"Error: {str(e)}", "failed")

@app.get("/api/jobs")
async def get_all_jobs():
    jobs_list = [{
        "id": job["id"],
        "status": job["status"],
        "video_name": job["video_name"],
        "created_at": job["created_at"].isoformat(),
        "updated_at": job["updated_at"].isoformat(),
        "progress": job["progress"],
        "current_step": job["current_step"],
        "ai_model": job.get("ai_model", ""),
        "translation_language": job.get("translation_language"),
        "output_files": job.get("output_files", {}),
        "error": job.get("error"),
    } for job in jobs_db.values()]
    return {"jobs": jobs_list}

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]
    return {
        "id": job["id"],
        "status": job["status"],
        "video_name": job["video_name"],
        "created_at": job["created_at"].isoformat(),
        "updated_at": job["updated_at"].isoformat(),
        "progress": job["progress"],
        "current_step": job["current_step"],
        "ai_model": job.get("ai_model", ""),
        "translation_language": job.get("translation_language"),
        "output_files": job.get("output_files", {}),
        "error": job.get("error"),
    }

@app.get("/api/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]
    if file_type not in job.get("output_files", {}):
        raise HTTPException(status_code=404, detail=f"File type {file_type} not available")

    file_path = job["output_files"][file_type]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=file_path, filename=os.path.basename(file_path))

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]
    if "process_dir" in job.get("output_files", {}):
        process_dir = job["output_files"]["process_dir"]
        if os.path.exists(process_dir):
            shutil.rmtree(process_dir)

    upload_dir = os.path.join(BASE_DIR, "uploads", job_id)
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)

    del jobs_db[job_id]
    return {"status": "deleted", "job_id": job_id}

@app.get("/api/keys/status")
async def get_api_key_status():
    keys = get_api_keys()
    return {"openai": bool(keys.get("openai")), "anthropic": bool(keys.get("anthropic"))}

@app.post("/api/keys/update")
async def update_api_keys(keys: ApiKeyUpdate):
    try:
        if keys.openai_key:
            save_api_keys(openai_api_key=keys.openai_key)
        if keys.anthropic_key:
            save_api_keys(anthropic_api_key=keys.anthropic_key)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    active_websockets[client_id] = websocket

    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_json({"echo": data}, client_id)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        if client_id in active_websockets:
            del active_websockets[client_id]

# Serve frontend static files
static_dir = os.path.join(BASE_DIR, "web_app", "dist")
if os.path.exists(static_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(static_dir, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            raise HTTPException(status_code=404)

        file_path = os.path.join(static_dir, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)

        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)

        raise HTTPException(status_code=404)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)