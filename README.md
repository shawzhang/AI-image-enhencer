# 🖼️ AI Image Enhancer

A professional-grade image enhancement web app powered by deep learning and advanced computer vision. Built with [FastAPI](https://fastapi.tiangolo.com/), [OpenCV](https://opencv.org/), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), and [GFPGAN](https://github.com/TencentARC/GFPGAN).

## ✨ Features

### 🔧 Restoration
| Feature | Technique | Description |
|---------|-----------|-------------|
| **Noise Reduction** | Non-Local Means (NLM) | Removes grain, dust, scratches, and JPEG compression artifacts from old/scanned photos |
| **Deblur / Clarity** | Gaussian Unsharp Mask | Sharpens blurry or out-of-focus areas without introducing noise |
| **Super-Resolution** | Real-ESRGAN (AI) | Upscales low-resolution images 2x or 4x, generating realistic textures (hair, skin, fabric) that were completely missing |
| **Face Restoration** | GFPGAN v1.4 (AI) | Detects and restores facial details — eyes, teeth, and skin — in old or blurry photos |

### 🎨 Enhancement
| Feature | Description |
|---------|-------------|
| **Sharpness** | Fine-tune edge sharpness on the luminance channel to avoid color artifacts |
| **Contrast (CLAHE)** | Adaptive local contrast enhancement that won't overexpose highlights or crush shadows |
| **Color Saturation** | Boost or reduce color vibrance (0 = grayscale, 1 = natural, 3 = vivid) |
| **Brightness** | Lighten or darken the overall image |
| **Color Temperature** | Shift the white balance from cool (blue) to warm (amber) |

### 🪄 Magic Actions
| Feature | Description |
|---------|-------------|
| **Auto Settings** | Intelligently analyzes image noise, blur, exposure, and faces to automatically configure optimal parameters |
| **Monet Style** | Applies an advanced Pyramid Mean Shift filtering pipeline to transform any photo into an impressionist oil painting |

### 📁 File Support
- **Input**: JPEG, PNG, HEIC/HEIF (Apple photos), and other common image formats
- **Output**: Save as JPEG or PNG
- **UI**: Drag-and-drop upload, before/after comparison slider

## 🏗️ Architecture

```
image_enhencement/
├── main.py          ← FastAPI server (REST API + static file serving)
├── engine.py        ← Image processing pipeline (UI-agnostic)
├── static/
│   ├── index.html   ← Web UI (single-page app)
│   ├── style.css    ← Premium dark-mode design system
│   └── app.js       ← Client-side logic
├── pyproject.toml   ← Project config & dependencies
└── README.md
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the web UI |
| `GET` | `/api/device` | Current compute device (MPS/CUDA/CPU) |
| `POST` | `/api/upload` | Upload image, returns preview + image ID |
| `POST` | `/api/analyze` | Analyzes image and returns recommended optimal parameters |
| `POST` | `/api/enhance` | Apply enhancement with parameters, returns base64 result |
| `POST` | `/api/download` | Download enhanced image as JPEG or PNG |

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Installation

```bash
git clone <repo-url>
cd image_enhencement
uv sync
```

> **Note:** The first `uv sync` will install PyTorch and supporting libraries (~300MB+). On Apple Silicon Macs, PyTorch automatically uses **MPS (Metal Performance Shaders)** for GPU-accelerated inference.

### Run

```bash
uv run python main.py
```

Then open **http://127.0.0.1:8000** in your browser.

> **First-time model downloads:** The first time you enable Super-Resolution or Face Restoration, model weights will be auto-downloaded (~65MB for Real-ESRGAN, ~350MB for GFPGAN). They are cached locally for subsequent use.

## 🖥️ Hardware Support

| Platform | Acceleration | Status |
|----------|-------------|--------|
| Apple Silicon (M1–M5) | MPS (Metal) | ✅ Fast |
| NVIDIA GPU | CUDA | ✅ Fast |
| CPU only | None | ✅ Works (slower for AI features) |

The app auto-detects the best available device at startup.

## 📸 Usage Tips

- **Auto Settings**: Click the 🤖 **Auto** button to analyze your image. The AI will detect noise, blur, exposure, and faces, then automatically configure the optimal sliders for you!
- **Monet Style**: Click the 🎨 **Monet** button to apply an algorithmic impressionist oil painting effect. This enhances colors and creates beautiful brush-stroke textures.
- **Old grainy photos**: Start with Noise Reduction (5–15), then add mild Sharpness (0.3–0.5) and Contrast (1.5–2.0)
- **Low-res images**: Toggle Super-Resolution with 2x for moderate upscaling, 4x for maximum detail
- **Blurry faces**: Enable Face Restoration — it works best on photos where faces are visible but degraded
- **Faded colors**: Increase Color Saturation (1.3–1.8) and adjust Color Temperature to correct yellow/blue casts
- **Quick adjustments**: Lightweight sliders auto-enhance with 300ms debounce after first manual enhance

## 🛠️ Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **Frontend**: Vanilla HTML/CSS/JS with dark-mode glassmorphism design
- **Image Processing**: [OpenCV](https://opencv.org/) (NLM denoising, CLAHE, unsharp masking, color space transforms)
- **Super-Resolution**: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (RRDBNet architecture)
- **Face Restoration**: [GFPGAN](https://github.com/TencentARC/GFPGAN) (v1.4, clean architecture)
- **Deep Learning**: [PyTorch](https://pytorch.org/) (MPS/CUDA/CPU)
- **HEIC Support**: [pillow-heif](https://github.com/bigcat88/pillow_heif)

## 📝 Known Issues

- **basicsr compatibility**: The `basicsr` package has a deprecated import (`torchvision.transforms.functional_tensor`) that must be patched after installation. If `uv sync` reinstalls `basicsr`, the patch in `.venv/lib/.../basicsr/data/degradations.py` (line 8) may need to be re-applied:
  ```
  # Change this:
  from torchvision.transforms.functional_tensor import rgb_to_grayscale
  # To this:
  from torchvision.transforms.functional import rgb_to_grayscale
  ```
- **GFPGAN on MPS**: GFPGAN runs on CPU fallback since some convolution ops aren't supported on Metal yet. Real-ESRGAN uses MPS normally.
