# Models Folder Map (do not commit weights)

Create these directories under your ComfyUI root:

- `models/checkpoints` — base/sdxl/etc. checkpoints
- `models/vae` — VAE files
- `models/clip` — CLIP text encoders
- `models/clip_vision` — CLIP vision encoders
- `models/unet` — UNet backbones
- `models/upscale_models` — ESRGAN/RealESRGAN/upscalers
- `models/loras` — LoRA adapters
- `models/embeddings` — Textual inversion embeddings

Keep large binaries out of git. Use Git LFS if you must track small pointers.
