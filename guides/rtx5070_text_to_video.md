# Photorealistic Text-to-Video AI Setup Guide for Indie Filmmakers

**Tested System:** NVIDIA GeForce RTX 5070 (12GB VRAM) · 32GB RAM · AMD Ryzen 7 · Windows 11

This guide walks through a production-ready pipeline for installing and running modern text-to-video models on hardware constrained to 12GB of VRAM. The recommendations prioritize photorealism, VRAM efficiency, and reliable tooling for indie filmmakers.

> **TL;DR:** CogVideoX-5B with FP8/INT8 quantization is the best day-to-day option on a 12GB RTX 5070. Stable Video Diffusion handles image-to-video workflows, AnimateDiff covers the Stable Diffusion ecosystem, and Mochi-1 is viable only with heavy optimization.

---

## 1. Model Selection Matrix

| Model | Photorealism | Speed (480p, 5s) | VRAM Usage | Max Resolution | Max Duration | 12GB Compatible | Setup Complexity |
|-------|--------------|------------------|------------|----------------|--------------|-----------------|------------------|
| **CogVideoX-5B** | 8/10 | 10–15 min | 4.4–12GB | 720p | 10s @ 16fps | ✅ Excellent | Medium |
| **Stable Video Diffusion** | 8/10 | 4–5 min | 8–11GB | 1024×576 | 4s @ 6fps | ✅ Excellent | Low |
| **AnimateDiff** | 7/10 | 2–3 min | 8–13GB | 512×768 | 16 frames | ✅ Good | Low |
| **Mochi-1** | 9/10 | 50–60 min | 12–22GB | 480p | 5.4s @ 30fps | ⚠️ Requires tuning | High |
| **HunyuanVideo** | 9/10 | 20–30 min | 16–24GB | 720p | 15s @ 24fps | ❌ Not feasible | High |
| **LTXVideo** | 8/10 | 15–18 sec | 6–12GB | 768×512 | 5s @ 24fps | ✅ Good | Medium |

**Workflow tip:** Generate at 480p–512p to stay within VRAM limits, then upscale to 1080p/4K using Topaz Video AI or RealESRGAN.

---

## 2. Critical Prerequisites for RTX 5070 (Blackwell)

1. **NVIDIA Driver ≥ 572.70**
   - Download: <https://www.nvidia.com/download/index.aspx>
   - Install with "Custom" → Graphics Driver + PhysX only → Clean installation.
   - Verify: `nvidia-smi` should show driver `572.70+` and CUDA `12.8`.

2. **CUDA Toolkit 12.8**
   - Download: <https://developer.nvidia.com/cuda-12-8-0-download-archive>
   - Install Toolkit only. Ensure `CUDA_PATH` and PATH entries point to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`.
   - Verify: `nvcc --version` → `release 12.8`.

3. **Python 3.12 (64-bit)**
   - Download: <https://www.python.org/downloads/windows/>
   - Install for all users, add to PATH, disable path length limit.
   - Upgrade tooling: `python -m pip install --upgrade pip wheel setuptools`.

4. **Git + Git LFS**
   - Download: <https://git-scm.com/download/win>
   - Run `git lfs install` and `git config --global core.longpaths true`.

5. **Windows Long Path Support**
   - Run elevated: `reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f`.

6. **Project Virtual Environment**
   ```powershell
   mkdir C:\AI\text-to-video
   cd C:\AI\text-to-video
   python -m venv venv
   venv\Scripts\activate
   ```

7. **PyTorch Nightly (Required)**
   ```powershell
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   ```
   Verify within Python:
   ```python
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   print(torch.version.cuda)
   print(torch.cuda.get_device_name(0))
   ```

8. **Common Libraries**
   ```powershell
   pip install diffusers transformers accelerate safetensors
   pip install omegaconf einops opencv-python pillow imageio imageio-ffmpeg
   pip install sentencepiece protobuf huggingface_hub
   ```

9. **xFormers (cu128 build)**
   ```powershell
   pip install xformers --index-url https://download.pytorch.org/whl/cu128
   python -m xformers.info
   ```

10. **Hugging Face CLI Login (optional but recommended)**
    ```powershell
    huggingface-cli login
    ```

---

## 3. ComfyUI Deployment

### Option A: Portable Build (Recommended)
1. Download from <https://github.com/comfyanonymous/ComfyUI/releases> (`ComfyUI_windows_portable.7z`).
2. Extract to `C:\ComfyUI`.
3. Launch with `run_nvidia_gpu.bat`.
4. Install ComfyUI Manager:
   ```powershell
   cd C:\ComfyUI\ComfyUI\custom_nodes
   git clone https://github.com/ltdrdata/ComfyUI-Manager.git
   ```

### Directory Layout
```
C:\ComfyUI\
├── models\
│   ├── checkpoints\
│   ├── diffusion_models\
│   ├── loras\
│   ├── text_encoders\
│   └── vae\
├── custom_nodes\
├── input\
└── output\
```

---

## 4. Model Playbooks

### 4.1 CogVideoX-5B (Primary Recommendation)
- **Best for:** 720p clips, strong text alignment, FP8 efficiency.
- **Key settings:** width 720 · height 480 · 33–49 frames · 50 steps · CFG 6.0.

**Install wrapper:**
```powershell
cd C:\ComfyUI\ComfyUI\custom_nodes
git clone https://github.com/kijai/ComfyUI-CogVideoXWrapper.git
cd ComfyUI-CogVideoXWrapper
pip install -r requirements.txt
```

**Download models:**
```powershell
mkdir C:\ComfyUI\models\CogVideoX
huggingface-cli download THUDM/CogVideoX-5b --local-dir C:\ComfyUI\models\CogVideoX\CogVideoX-5b
huggingface-cli download THUDM/CogVideoX-5b-I2V --local-dir C:\ComfyUI\models\CogVideoX\CogVideoX-5b-I2V
huggingface-cli download THUDM/CogVideoX-5b --include "text_encoder/*" --local-dir C:\ComfyUI\models\text_encoders
```
Use FP8 precision, enable VAE tiling and (optionally) CPU offload. Expect 10–15 minutes per 6-second clip.

**Sample prompt:**
```
A professional filmmaker operating a cinema camera on a stabilizer, tracking shot through a modern art gallery with dramatic lighting, golden hour, cinematic color grading, 35mm anamorphic lens
```

### 4.2 Stable Video Diffusion (SVD-XT)
- **Best for:** Animating still images with minimal VRAM impact.
- **Limitations:** Image-to-video only, 25 frames.

**Download:**
```powershell
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir C:\ComfyUI\models\checkpoints\svd-xt
```

**Python snippet:**
```python
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import torch

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

image = load_image("input.png").resize((1024, 576))
frames = pipe(
    image,
    decode_chunk_size=8,
    motion_bucket_id=127,
    noise_aug_strength=0.02,
    num_frames=25,
    generator=torch.manual_seed(42),
).frames[0]
export_to_video(frames, "output.mp4", fps=7)
```

### 4.3 AnimateDiff
- **Best for:** Users with Stable Diffusion expertise, extensive LoRA support.
- **Resolution:** 512×512 or 512×768, 16–24 frames.

**Install nodes + motion modules:**
```powershell
cd C:\ComfyUI\ComfyUI\custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git
cd ComfyUI-AnimateDiff-Evolved
pip install -r requirements.txt
```
Download motion modules from <https://huggingface.co/guoyww/animatediff/tree/main> (e.g., `mm_sd_v15_v2.ckpt`) into `models` subfolder.

**Base model example:**
```powershell
huggingface-cli download SG161222/Realistic_Vision_V5.1_noVAE --include "*.safetensors" --local-dir C:\ComfyUI\models\checkpoints\realistic-vision
```

**Prompt pair:**
- Positive: `masterpiece, cinematic shot of a woman walking down a rainy city street at night, neon reflections, volumetric lighting`
- Negative: `worst quality, static image, blurry, distorted`

### 4.4 Mochi-1 (Advanced)
- **Photorealism leader but VRAM heavy.** Use only with Q5 GGUF quantization and aggressive offloading.

**Install wrapper + model:**
```powershell
cd C:\ComfyUI\ComfyUI\custom_nodes
git clone https://github.com/kijai/ComfyUI-MochiWrapper.git
cd ComfyUI-MochiWrapper
pip install -r requirements.txt
huggingface-cli download city96/mochi-1-preview-gguf --include "*Q5*.gguf" --local-dir C:\ComfyUI\models\diffusion_models\mochi
huggingface-cli download genmo/mochi-1-preview --include "text_encoder/*" --local-dir C:\ComfyUI\models\text_encoders\mochi
huggingface-cli download genmo/mochi-1-preview --include "vae/*" --local-dir C:\ComfyUI\models\vae\mochi
```

**Settings for 12GB:** 640×480, ≤61 frames, 50–100 steps, enable VAE tiling + CPU offload. Expect 50–60 minutes per clip.

---

## 5. VRAM Optimization Playbook

- **Quantization:** Prefer FP8 or INT8 for transformers; keep VAEs in FP16 for quality.
- **xFormers:** `pipe.enable_xformers_memory_efficient_attention()` (≈30–40% VRAM savings).
- **Attention slicing:** `pipe.enable_attention_slicing("auto")` when OOM occurs.
- **CPU offload:** `pipe.enable_model_cpu_offload()` or `pipe.enable_sequential_cpu_offload()` (needs ≥32GB RAM).
- **VAE tiling/slicing:** Required for >512p resolutions.
- **torch.compile:** `pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")` for ~10–15% speedup after warm-up.

**Two-stage workflow:**
1. Generate at 480p.
2. Export frames to PNG.
3. Upscale via Topaz Video AI or `realesrgan-ncnn-vulkan -i input.mp4 -o output.mp4 -s 2`.

---

## 6. Troubleshooting Cheatsheet

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | Close other GPU apps → reduce resolution/frames → enable CPU offload → rerun with attention slicing.
| `torch.cuda.is_available() == False` | Verify driver 572.70+ → reinstall PyTorch nightly (cu128) → ensure no CPU-only wheels are installed.
| `sm_120 not supported` | You installed stable PyTorch; reinstall nightly with `--pre` + `cu128` index.
| Slow generations | Confirm xFormers installed → enable `torch.compile` → monitor temps via `nvidia-smi -l 1`.
| CogVideoX allocation error | Retry immediately or call `torch.cuda.empty_cache()` before second run.
| Black screen post-driver install | Use DDU in Safe Mode, reinstall driver 572.70+ (avoid 572.60–572.65).
| Long path errors | Enable Windows long paths + Git `core.longpaths`.

---

## 7. Production Workflow Tips

- Generate 10–20 low-res variants, curate best takes, then upscale.
- Budget 10–15 minutes per CogVideoX clip, 4–5 minutes for SVD, 50–60 minutes for Mochi-1 hero shots.
- Keep 10GB+ free system RAM when using CPU offload.
- Monitor VRAM with `nvidia-smi` before launches; close browsers/streaming apps.

**Multi-model strategy:** Use CogVideoX for primary generation, SVD for animating boards, AnimateDiff for SD ecosystems, Mochi-1 for marquee shots.

---

## 8. Reference Links

- CogVideoX: <https://github.com/THUDM/CogVideo>
- Mochi-1: <https://github.com/genmoai/mochi>
- ComfyUI documentation: <https://docs.comfy.org/>
- Diffusers docs: <https://huggingface.co/docs/diffusers>
- CivitAI: <https://civitai.com/>
- Hugging Face models: <https://huggingface.co/models?pipeline_tag=text-to-video>

---

**Guide version:** November 2025 · Last verified with RTX 5070 driver 572.70, CUDA 12.8, PyTorch nightly (cu128)
