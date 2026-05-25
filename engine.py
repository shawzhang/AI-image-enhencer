"""
Image Enhancement Engine
========================
Core image processing pipeline with deep learning super-resolution (Real-ESRGAN),
face restoration (GFPGAN), and advanced OpenCV filters.

This module is UI-agnostic — it can be used with any web framework or CLI.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from pillow_heif import register_heif_opener

# Enable HEIC/HEIF support
register_heif_opener()

# Raise PIL's pixel limit to handle large images after 4x upscaling
Image.MAX_IMAGE_PIXELS = 500_000_000

# ---------------------------------------------------------------------------
# Device detection: prefer MPS (Apple Silicon) > CUDA > CPU
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DEVICE_NAME = "Apple MPS (Metal)"
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = "NVIDIA CUDA"
else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"

print(f"[Engine] Using device: {DEVICE_NAME}")

# ---------------------------------------------------------------------------
# Lazy-loaded deep learning models
# ---------------------------------------------------------------------------
_esrgan_models = {}
_gfpgan_restorer = None
_gfpgan_available = True


def _get_esrgan_model(scale: int):
    """Lazy-load Real-ESRGAN model for the given scale (2 or 4)."""
    global _esrgan_models
    if scale in _esrgan_models:
        return _esrgan_models[scale]

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    if scale == 2:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_url,
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=10,
        half=False,
        device=DEVICE,
    )
    _esrgan_models[scale] = upsampler
    print(f"[Real-ESRGAN] Loaded {scale}x model on {DEVICE_NAME}")
    return upsampler


def _get_gfpgan_restorer():
    """Lazy-load GFPGAN model. Returns None if unavailable."""
    global _gfpgan_restorer, _gfpgan_available
    if not _gfpgan_available:
        return None
    if _gfpgan_restorer is not None:
        return _gfpgan_restorer

    try:
        from gfpgan import GFPGANer

        gfpgan_device = torch.device("cpu")
        restorer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=gfpgan_device,
        )
        _gfpgan_restorer = restorer
        print(f"[GFPGAN] Loaded face restoration model (CPU fallback)")
        return restorer
    except Exception as e:
        print(f"[GFPGAN] Failed to load: {e}")
        _gfpgan_available = False
        return None


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_image_file(file_path: str) -> np.ndarray:
    """Load an image file (including HEIC) and return as RGB numpy array."""
    img = Image.open(file_path)
    img = img.convert("RGB")
    return np.array(img)


# ---------------------------------------------------------------------------
# Enhancement pipeline
# ---------------------------------------------------------------------------
def enhance_image(
    img: np.ndarray,
    noise_reduction: int = 0,
    deblur_strength: float = 0.0,
    enable_super_res: bool = False,
    super_res_scale: str = "2x",
    enable_face_restore: bool = False,
    sharpness: float = 0.5,
    contrast: float = 1.0,
    color_saturation: float = 1.0,
    brightness: int = 0,
    color_temp: int = 0,
    monet_style: bool = False,
) -> np.ndarray:
    """
    Apply the full enhancement pipeline to an RGB numpy array.
    Returns the enhanced RGB numpy array.
    """
    # Ensure 3-channel RGB
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # ── Step 1: Noise Reduction (Non-Local Means) ──────────────────────────
    if noise_reduction > 0:
        h = int(noise_reduction)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, 7, 21)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Step 2: Deblur / Clarity (Gaussian Unsharp Mask) ───────────────────
    if deblur_strength > 0:
        img_float = img.astype(np.float32)
        blurred = cv2.GaussianBlur(img_float, (0, 0), sigmaX=3.0)
        img_float = cv2.addWeighted(img_float, 1.0 + deblur_strength,
                                    blurred, -deblur_strength, 0)
        img = np.clip(img_float, 0, 255).astype(np.uint8)

    # ── Step 3: Super-Resolution (Real-ESRGAN) ────────────────────────────
    if enable_super_res:
        try:
            scale = 2 if super_res_scale == "2x" else 4
            upsampler = _get_esrgan_model(scale)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            output_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)
            img = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[Real-ESRGAN] Error: {e}")

    # ── Step 4: Face Restoration (GFPGAN) ─────────────────────────────────
    if enable_face_restore:
        restorer = _get_gfpgan_restorer()
        if restorer is not None:
            try:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                _, _, restored_bgr = restorer.enhance(
                    img_bgr,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )
                img = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"[GFPGAN] Error: {e}")

    # ── Step 5–7: Sharpness, Contrast, Saturation (YCrCb space) ───────────
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_yuv)

    # Contrast (CLAHE on luminance)
    clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)

    # Sharpness (unsharp mask on luminance)
    y_float = y_eq.astype(np.float32)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    y_sharp_base = cv2.filter2D(y_float, -1, kernel)
    y_result = cv2.addWeighted(y_sharp_base, sharpness, y_float,
                               1.0 - sharpness, 0)
    y_result = np.clip(y_result, 0, 255).astype(np.uint8)

    # Color Saturation
    if color_saturation != 1.0:
        cr = np.clip((cr.astype(np.float32) - 128.0) * color_saturation + 128.0,
                     0, 255).astype(np.uint8)
        cb = np.clip((cb.astype(np.float32) - 128.0) * color_saturation + 128.0,
                     0, 255).astype(np.uint8)

    merged = cv2.merge([y_result, cr, cb])
    img = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)

    # ── Step 8: Brightness ────────────────────────────────────────────────
    if brightness != 0:
        img = np.clip(img.astype(np.float32) + brightness, 0, 255).astype(np.uint8)

    # ── Step 9: Color Temperature ─────────────────────────────────────────
    if color_temp != 0:
        img_float = img.astype(np.float32)
        img_float[:, :, 0] = np.clip(img_float[:, :, 0] + color_temp, 0, 255)  # R
        img_float[:, :, 2] = np.clip(img_float[:, :, 2] - color_temp, 0, 255)  # B
        img = img_float.astype(np.uint8)

    # ── Step 10: Monet Oil Painting Effect ─────────────────────────────────
    if monet_style:
        # 1. Smooth out details but preserve edges (Oil paint strokes)
        # pyrMeanShiftFiltering gives a highly characteristic 'painted' flat-region look
        # It can be slow on huge images, so we resize if needed, filter, then resize back
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > 1200:
            scale = 1200 / max(h, w)
            work_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            work_img = img.copy()

        # Apply mean shift filtering
        painted = cv2.pyrMeanShiftFiltering(work_img, sp=12, sr=35)
        
        # 2. Add slight median blur to blend the 'strokes' together
        painted = cv2.medianBlur(painted, 3)

        # 3. Enhance colors slightly (Monet impressionist style)
        hsv = cv2.cvtColor(painted, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255) # boost saturation
        painted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 4. Restore original size if scaled down
        if scale != 1.0:
            img = cv2.resize(painted, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            img = painted

    return img


# ---------------------------------------------------------------------------
# Image analysis — auto-detect optimal parameters
# ---------------------------------------------------------------------------
def analyze_image(img: np.ndarray) -> dict:
    """
    Analyze an RGB image and return recommended enhancement parameters.

    Examines:
    - Noise level (Laplacian variance on small patches)
    - Blur level (global Laplacian variance)
    - Brightness (mean luminance)
    - Contrast (std dev of luminance)
    - Color saturation (mean chrominance deviation)
    - Resolution (total pixels)
    - Face presence (Haar cascade detector)

    Returns a dict with recommended parameter values + analysis summary.
    """
    h, w = img.shape[:2]
    total_pixels = h * w

    # Work on a downscaled copy for speed (cap at 800px wide)
    if w > 800:
        scale = 800 / w
        analysis_img = cv2.resize(img, (800, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        analysis_img = img.copy()

    # Convert to different color spaces
    gray = cv2.cvtColor(analysis_img, cv2.COLOR_RGB2GRAY)
    yuv = cv2.cvtColor(analysis_img, cv2.COLOR_RGB2YCrCb)
    y_ch, cr_ch, cb_ch = cv2.split(yuv)

    # ── Blur detection (Laplacian variance) ────────────────────────────────
    # Low value = blurry image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # ── Noise estimation (high-frequency energy in small patches) ──────────
    # Compute Laplacian, then measure variance in local patches
    # High noise → high local variance in flat regions
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    # Estimate noise as median absolute deviation of Laplacian
    noise_sigma = np.median(np.abs(lap)) * 1.4826  # MAD estimator

    # ── Brightness (mean luminance) ────────────────────────────────────────
    mean_brightness = float(np.mean(y_ch))

    # ── Contrast (std dev of luminance) ────────────────────────────────────
    contrast_std = float(np.std(y_ch))

    # ── Color saturation (mean chrominance deviation from neutral 128) ─────
    cr_dev = np.mean(np.abs(cr_ch.astype(np.float32) - 128.0))
    cb_dev = np.mean(np.abs(cb_ch.astype(np.float32) - 128.0))
    mean_saturation = float((cr_dev + cb_dev) / 2.0)

    # ── Face detection (Haar cascade — fast) ───────────────────────────────
    has_faces = False
    face_count = 0
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_count = len(faces)
        has_faces = face_count > 0
    except Exception:
        pass

    # ── Build recommended parameters ───────────────────────────────────────
    params = {
        "noise_reduction": 0,
        "deblur_strength": 0.0,
        "enable_super_res": False,
        "super_res_scale": "2x",
        "enable_face_restore": False,
        "sharpness": 0.5,
        "contrast": 1.0,
        "color_saturation": 1.0,
        "brightness": 0,
        "color_temp": 0,
    }

    hints = []

    # Noise recommendation
    if noise_sigma > 15:
        params["noise_reduction"] = min(int(noise_sigma * 0.6), 25)
        hints.append(f"High noise detected (σ={noise_sigma:.0f}) → denoising {params['noise_reduction']}")
    elif noise_sigma > 8:
        params["noise_reduction"] = min(int(noise_sigma * 0.4), 15)
        hints.append(f"Moderate noise detected (σ={noise_sigma:.0f}) → light denoising {params['noise_reduction']}")

    # Blur recommendation
    if laplacian_var < 50:
        params["deblur_strength"] = 2.0
        params["sharpness"] = 1.0
        hints.append(f"Very blurry image (sharpness={laplacian_var:.0f}) → strong deblur + sharpening")
    elif laplacian_var < 200:
        params["deblur_strength"] = 0.8
        params["sharpness"] = 0.7
        hints.append(f"Slightly soft image (sharpness={laplacian_var:.0f}) → mild deblur")

    # Brightness recommendation
    if mean_brightness < 80:
        params["brightness"] = min(int((120 - mean_brightness) * 0.5), 40)
        hints.append(f"Dark image (mean={mean_brightness:.0f}) → brightness +{params['brightness']}")
    elif mean_brightness > 200:
        params["brightness"] = max(int((140 - mean_brightness) * 0.4), -40)
        hints.append(f"Overexposed image (mean={mean_brightness:.0f}) → brightness {params['brightness']}")

    # Contrast recommendation
    if contrast_std < 35:
        params["contrast"] = 2.5
        hints.append(f"Low contrast (std={contrast_std:.0f}) → CLAHE boost to 2.5")
    elif contrast_std < 50:
        params["contrast"] = 1.5
        hints.append(f"Moderate contrast (std={contrast_std:.0f}) → mild CLAHE 1.5")

    # Saturation recommendation
    if mean_saturation < 8:
        params["color_saturation"] = 1.6
        hints.append(f"Desaturated/faded colors (sat={mean_saturation:.1f}) → saturation boost 1.6")
    elif mean_saturation < 15:
        params["color_saturation"] = 1.2
        hints.append(f"Slightly muted colors (sat={mean_saturation:.1f}) → mild saturation 1.2")

    # Resolution recommendation
    if total_pixels < 500_000:  # less than ~700x700
        params["enable_super_res"] = True
        params["super_res_scale"] = "4x"
        hints.append(f"Very low resolution ({w}×{h}) → 4x super-resolution recommended")
    elif total_pixels < 2_000_000:  # less than ~1400x1400
        params["enable_super_res"] = True
        params["super_res_scale"] = "2x"
        hints.append(f"Low resolution ({w}×{h}) → 2x super-resolution recommended")

    # Face recommendation
    if has_faces:
        params["enable_face_restore"] = True
        hints.append(f"Detected {face_count} face(s) → face restoration enabled")

    if not hints:
        hints.append("Image looks good! Only minor adjustments applied.")

    return {
        "params": params,
        "analysis": {
            "noise_sigma": round(noise_sigma, 1),
            "blur_score": round(laplacian_var, 1),
            "mean_brightness": round(mean_brightness, 1),
            "contrast_std": round(contrast_std, 1),
            "mean_saturation": round(mean_saturation, 1),
            "resolution": f"{w}×{h}",
            "total_pixels": total_pixels,
            "face_count": face_count,
        },
        "hints": hints,
    }
