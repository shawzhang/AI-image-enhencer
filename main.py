"""
AI Image Enhancer — FastAPI Backend
====================================
REST API for the image enhancement pipeline.
Serves the web UI and provides endpoints for image upload, enhancement, and download.
"""

import uuid
import io
import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import engine

# ---------------------------------------------------------------------------
# In-memory image store (keyed by session UUID)
# ---------------------------------------------------------------------------
_image_store: dict[str, np.ndarray] = {}
_enhanced_store: dict[str, np.ndarray] = {}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="AI Image Enhancer")

# Serve static files (CSS, JS)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class EnhanceRequest(BaseModel):
    image_id: str
    noise_reduction: int = 0
    deblur_strength: float = 0.0
    enable_super_res: bool = False
    super_res_scale: str = "2x"
    enable_face_restore: bool = False
    sharpness: float = 0.5
    contrast: float = 1.0
    color_saturation: float = 1.0
    brightness: int = 0
    color_temp: int = 0
    monet_style: bool = False


class DownloadRequest(BaseModel):
    image_id: str
    format: str = "jpeg"


class AnalyzeRequest(BaseModel):
    image_id: str


# ---------------------------------------------------------------------------
# Helper: numpy RGB → base64 data URI
# ---------------------------------------------------------------------------
def _img_to_data_uri(img: np.ndarray, fmt: str = "jpeg", quality: int = 85) -> str:
    """Convert an RGB numpy array to a base64 data URI string."""
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    if fmt == "jpeg":
        pil_img.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        pil_img.save(buf, format="PNG")
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main web UI."""
    index_path = static_dir / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/device")
async def get_device():
    """Return the current compute device."""
    return {"device": engine.DEVICE_NAME}


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file. Returns an image_id and a preview data URI.
    Supports JPEG, PNG, HEIC, and other PIL-compatible formats.
    """
    # Save to temp file for PIL/HEIC loading
    import tempfile
    suffix = Path(file.filename or "image.jpg").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        img = engine.load_image_file(tmp_path)
    finally:
        import os
        os.unlink(tmp_path)

    # Store original
    image_id = str(uuid.uuid4())
    _image_store[image_id] = img

    # Generate preview (cap at 1200px wide for fast transfer)
    h, w = img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        preview = cv2.resize(img, (1200, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        preview = img

    preview_uri = _img_to_data_uri(preview, "jpeg", quality=80)

    return {
        "image_id": image_id,
        "preview": preview_uri,
        "width": w,
        "height": h,
    }


@app.post("/api/enhance")
async def enhance(req: EnhanceRequest):
    """Apply enhancement pipeline and return the result as a base64 data URI."""
    if req.image_id not in _image_store:
        return {"error": "Image not found. Please upload again."}

    original = _image_store[req.image_id]

    enhanced = engine.enhance_image(
        original.copy(),
        noise_reduction=req.noise_reduction,
        deblur_strength=req.deblur_strength,
        enable_super_res=req.enable_super_res,
        super_res_scale=req.super_res_scale,
        enable_face_restore=req.enable_face_restore,
        sharpness=req.sharpness,
        contrast=req.contrast,
        color_saturation=req.color_saturation,
        brightness=req.brightness,
        color_temp=req.color_temp,
        monet_style=req.monet_style,
    )

    # Store enhanced for download
    _enhanced_store[req.image_id] = enhanced

    # Generate preview (cap size for fast transfer)
    h, w = enhanced.shape[:2]
    if w > 1600:
        scale = 1600 / w
        preview = cv2.resize(enhanced, (1600, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        preview = enhanced

    return {"image": _img_to_data_uri(preview, "jpeg", quality=85)}


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """Analyze the image and return recommended parameters."""
    if req.image_id not in _image_store:
        return {"error": "Image not found."}

    original = _image_store[req.image_id]
    result = engine.analyze_image(original)
    return result


@app.post("/api/download")
async def download(req: DownloadRequest):
    """Download the enhanced image as JPEG or PNG."""
    img = _enhanced_store.get(req.image_id)
    if img is None:
        img = _image_store.get(req.image_id)
    if img is None:
        return {"error": "Image not found."}

    pil_img = Image.fromarray(img)
    buf = io.BytesIO()

    if req.format.lower() == "png":
        pil_img.save(buf, format="PNG")
        media_type = "image/png"
        ext = "png"
    else:
        pil_img.convert("RGB").save(buf, format="JPEG", quality=95)
        media_type = "image/jpeg"
        ext = "jpeg"

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="enhanced.{ext}"'},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n🖼️  AI Image Enhancer")
    print(f"   Device: {engine.DEVICE_NAME}")
    print(f"   Open http://127.0.0.1:8000 in your browser\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
