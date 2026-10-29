"""FastAPI backend for Property Studio AI.

Provides REST API endpoints for video generation.
"""

from pathlib import Path
from typing import List, Optional
import shutil
import tempfile
import json
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import existing video generation modules
from src.validators import validate_property_input, PropertyImage, ValidationError
from src.image_classifier import detect_image_category, DISPLAY_NAMES
from src.script_generator import generate_script
from src.video_generator import generate_property_video, VideoGenerationError
from src.audio_generator import generate_tts_audio

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Property Studio AI API", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="assets"), name="static")


# Pydantic models
class PropertyRequest(BaseModel):
    name: str
    location: str
    price: str
    property_type: str
    bedrooms: int
    description: str
    features: List[str]
    call_to_action: str
    include_narration: bool = True
    include_background_music: bool = True
    tts_tld: str = "co.in"
    tts_slow: bool = False
    tts_engine: str = "auto"
    tts_gender: str = "female"
    bg_music_volume: float = 0.15
    sentence_gap: float = 0.3
    ending_hold: float = 0.8
    use_llm: bool = False
    api_key: Optional[str] = None


class ImageCategory(BaseModel):
    filename: str
    category: str


class GenerationRequest(BaseModel):
    property_data: PropertyRequest
    image_categories: List[ImageCategory]


class SampleData(BaseModel):
    preset: str


# Global progress tracking
generation_progress = {}


@app.get("/")
async def root():
    """Serve the HTML frontend."""
    from fastapi.responses import HTMLResponse
    html_path = PROJECT_ROOT / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return {"message": "Property Studio AI API", "version": "2.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/upload-images")
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload property images and return their paths."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if len(files) < 4 or len(files) > 8:
        raise HTTPException(status_code=400, detail="Please upload 4 to 8 images")

    uploaded_paths = []
    for file in files:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")

        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        uploaded_paths.append(str(file_path))

    return {"paths": uploaded_paths, "count": len(uploaded_paths)}


@app.post("/detect-categories")
async def detect_image_categories(image_paths: List[str]):
    """Detect image categories for uploaded property images."""
    categories = []
    for img_path in image_paths:
        try:
            from PIL import Image
            img = Image.open(img_path)
            _, detected_disp, conf = detect_image_category(img_path)
            categories.append({
                "path": img_path,
                "detected_category": detected_disp,
                "confidence": f"{conf:.0%}"
            })
        except Exception as e:
            categories.append({
                "path": img_path,
                "detected_category": "Auto-Detect",
                "confidence": "0%",
                "error": str(e)
            })

    return {"categories": categories}


@app.post("/api/generate")
async def generate_video_api(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    location: str = Form(...),
    price: str = Form(...),
    property_type: str = Form(...),
    call_to_action: str = Form(...),
    description: str = Form(""),
    bedrooms: int = Form(0),
    include_narration: bool = Form(True),
    include_background_music: bool = Form(True),
    tts_tld: str = Form("co.in"),
    tts_slow: bool = Form(False),
    bg_music_volume: float = Form(0.15),
    use_llm: bool = Form(False),
    api_key: Optional[str] = Form(None),
    use_sample: str = Form(""),
    features: List[str] = Form([]),
    images: List[UploadFile] = File([])
):
    """Generate property video with streaming progress updates."""
    # Generate unique job ID
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    generation_progress[job_id] = {"status": "queued", "progress": 0, "message": "Job queued"}

    # Handle sample images or uploaded images
    image_paths = []
    if use_sample:
        # Use sample images
        if use_sample == "green_valley":
            sample_dir = PROJECT_ROOT / "assets" / "sample_images"
        elif use_sample == "ocean_crest":
            sample_dir = PROJECT_ROOT / "assets" / "sample_images_villa"
        else:
            sample_dir = PROJECT_ROOT / "assets" / "sample_images"

        if sample_dir.exists():
            image_paths = sorted(list(sample_dir.glob("*.jpg")))
    else:
        # Handle uploaded images
        if not images:
            raise HTTPException(status_code=400, detail="No images uploaded")

        if len(images) < 4 or len(images) > 8:
            raise HTTPException(status_code=400, detail="Please upload 4 to 8 images")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_subdir = UPLOAD_DIR / timestamp
        upload_subdir.mkdir(parents=True, exist_ok=True)

        for idx, file in enumerate(images):
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")

            safe_filename = f"{idx}_{file.filename}"
            file_path = upload_subdir / safe_filename

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            image_paths.append(file_path)

    # Create PropertyImage objects with auto-detect category
    property_image_objects = []
    for img_path in image_paths:
        property_image_objects.append(PropertyImage(path=img_path, category=None))

    # Validate property input
    try:
        prop_input = validate_property_input(
            name=name,
            location=location,
            price=price,
            property_type=property_type,
            features=features,
            call_to_action=call_to_action,
            image_paths=property_image_objects,
            bedrooms=bedrooms if bedrooms > 0 else None,
            description=description
        )
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    # Create property data object
    from src.audio_generator import generate_scene_narration
    prop_data_dict = {
        "include_narration": include_narration,
        "include_background_music": include_background_music,
        "tts_tld": tts_tld,
        "tts_slow": tts_slow,
        "tts_engine": "auto",
        "tts_gender": "female",
        "bg_music_volume": bg_music_volume,
        "sentence_gap": 0.3,
        "ending_hold": 0.8,
        "use_llm": use_llm,
        "api_key": api_key
    }

    # Add background task
    background_tasks.add_task(
        run_video_generation,
        job_id,
        prop_input,
        prop_data_dict,
        OUTPUT_DIR
    )

    # Return streaming response
    async def event_stream():
        while True:
            if job_id not in generation_progress:
                yield f"{{\"error\": \"Job not found\"}}\n"
                break

            data = generation_progress[job_id]
            yield json.dumps(data) + "\n"

            if data["status"] in ["completed", "failed"]:
                break

            await asyncio.sleep(0.5)

    import asyncio
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


async def run_video_generation(job_id: str, prop_input, prop_data_dict, output_dir: Path):
    """Background task to generate video."""
    try:
        generation_progress[job_id] = {"status": "generating", "progress": 0.05, "message": "Initializing..."}

        def progress_callback(ratio: float, msg: str):
            generation_progress[job_id] = {
                "status": "generating",
                "progress": int(ratio * 100),
                "message": msg
            }

        # Generate script
        script = generate_script(prop_input, use_llm=prop_data_dict.get("use_llm", False), api_key=prop_data_dict.get("api_key"))
        progress_callback(0.10, "Script generated")

        # Generate video
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in prop_input.name if c.isalnum() or c in (' ', '_', '-')).rstrip()
        safe_name = safe_name.replace(" ", "_").lower()
        output_filename = f"{safe_name}_{timestamp_str}.mp4"
        output_path = output_dir / output_filename

        result_path = generate_property_video(
            prop=prop_input,
            output_path=output_path,
            script=script,
            include_narration=prop_data_dict.get("include_narration", True),
            include_background_music=prop_data_dict.get("include_background_music", True),
            fps=24,
            progress_callback=progress_callback,
            tts_tld=prop_data_dict.get("tts_tld", "co.in"),
            tts_slow=prop_data_dict.get("tts_slow", False),
            tts_engine=prop_data_dict.get("tts_engine", "auto"),
            tts_gender=prop_data_dict.get("tts_gender", "female"),
            bg_music_volume=prop_data_dict.get("bg_music_volume", 0.15),
            sentence_gap=prop_data_dict.get("sentence_gap", 0.3),
            ending_hold=prop_data_dict.get("ending_hold", 0.8)
        )

        generation_progress[job_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Video generated successfully",
            "done": True,
            "video_url": f"/download/{result_path.name}",
            "filename": result_path.name,
            "script": {
                "scenes": [{"headline": s.headline, "category_tag": s.category_tag, "duration": s.duration} for s in script.scenes]
            }
        }

    except Exception as e:
        generation_progress[job_id] = {
            "status": "failed",
            "progress": 0,
            "error": str(e)
        }


@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    """Get generation progress for a job."""
    if job_id not in generation_progress:
        raise HTTPException(status_code=404, detail="Job not found")

    return generation_progress[job_id]


@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download generated video."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=filename
    )


@app.get("/videos")
async def list_videos():
    """List all generated videos."""
    videos = []
    for file_path in OUTPUT_DIR.glob("*.mp4"):
        stat = file_path.stat()
        videos.append({
            "filename": file_path.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
        })

    return {"videos": sorted(videos, key=lambda x: x["created"], reverse=True)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
