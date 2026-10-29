"""HTTP backend for the Property Studio AI frontend."""

from pathlib import Path
import shutil
import tempfile
import threading
import time
from queue import Empty, Queue
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.script_generator import generate_script
from src.validators import PropertyImage, ValidationError, validate_property_input
from src.video_generator import VideoGenerationError, generate_property_video


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PRESETS = {
    "green_valley": PROJECT_ROOT / "assets" / "sample_images",
    "ocean_crest": PROJECT_ROOT / "assets" / "sample_images_villa",
}

app = FastAPI(title="Property Studio AI API")


def _sample_images(preset: str) -> List[Path]:
    directory = SAMPLE_PRESETS.get(preset)
    if directory is None or not directory.exists():
        raise HTTPException(status_code=400, detail="Unknown sample property preset.")
    images = sorted(directory.glob("*"))
    if not images:
        raise HTTPException(status_code=400, detail="The selected sample preset has no images.")
    return images


def _script_payload(script) -> dict:
    return script.to_dict()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/output/{filename}")
def download_output(filename: str):
    path = (OUTPUT_DIR / filename).resolve()
    if path.parent != OUTPUT_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.post("/api/generate")
async def generate(
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
    api_key: str = Form(""),
    use_sample: str = Form(""),
    features: List[str] = Form([]),
    images: Optional[List[UploadFile]] = File(None),
):
    image_paths: List[Path] = []
    upload_dir: Optional[Path] = None

    if images:
        upload_dir = Path(tempfile.mkdtemp(prefix="property_upload_"))
        for index, image in enumerate(images):
            suffix = Path(image.filename or "").suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                raise HTTPException(status_code=400, detail=f"Unsupported image format: {suffix or 'unknown'}")
            target = upload_dir / f"image_{index}{suffix}"
            with target.open("wb") as output_file:
                shutil.copyfileobj(image.file, output_file)
            image_paths.append(target)
    elif use_sample:
        image_paths = _sample_images(use_sample)
    else:
        raise HTTPException(status_code=400, detail="Upload images or choose a sample preset.")

    try:
        property_images = [PropertyImage(path=path) for path in image_paths]
        prop = validate_property_input(
            name=name,
            location=location,
            price=price,
            property_type=property_type,
            features=features,
            call_to_action=call_to_action,
            image_paths=property_images,
            bedrooms=bedrooms if bedrooms > 0 else None,
            description=description,
        )
    except ValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    script = generate_script(prop, use_llm=use_llm, api_key=api_key or None)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(char for char in prop.name if char.isalnum() or char in " _-").strip()
    filename = f"{safe_name.replace(' ', '_').lower()}_{timestamp}.mp4"
    output_path = OUTPUT_DIR / filename

    def render_stream():
        events = Queue()

        def progress(ratio: float, status: str):
            events.put({"progress": ratio, "status": status})

        yield '{"progress":0.05,"status":"Preparing validated property data..."}\n'

        def render_video():
            try:
                progress(0.12, "Generating cinematic script...")
                progress(0.18, "Rendering scenes and audio...")
                generate_property_video(
                    prop=prop,
                    output_path=output_path,
                    script=script,
                    include_narration=include_narration,
                    include_background_music=include_background_music,
                    fps=24,
                    progress_callback=progress,
                    tts_tld=tts_tld,
                    tts_slow=tts_slow,
                    bg_music_volume=max(0.0, min(bg_music_volume, 1.0)),
                )
                events.put({
                    "progress": 1,
                    "status": "Video ready!",
                    "done": True,
                    "video_url": f"/output/{filename}",
                    "filename": filename,
                    "script": _script_payload(script),
                })
            except (VideoGenerationError, OSError, ValueError) as error:
                events.put({"error": str(error)})

        worker = threading.Thread(target=render_video, daemon=True)
        worker.start()

        try:
            while worker.is_alive() or not events.empty():
                try:
                    import json
                    event = events.get(timeout=0.25)
                    yield json.dumps(event) + "\n"
                    if event.get("done") or event.get("error"):
                        break
                except Empty:
                    continue
        finally:
            worker.join()
            if upload_dir:
                shutil.rmtree(upload_dir, ignore_errors=True)

    return StreamingResponse(render_stream(), media_type="application/x-ndjson")


app.mount("/", StaticFiles(directory=PROJECT_ROOT, html=True), name="frontend")
