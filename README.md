# RVM-rtmp-pipeline

Real-time RTMP ingest-process-rebroadcast pipeline powered by Robust Video Matting (RVM).

`RVM-rtmp-pipeline` takes a live RTMP input stream, extracts human alpha/foreground with RVM, renders a privacy avatar (black silhouette today), and pushes the processed stream back to RTMP.

## Features

- RTMP input to RTMP output pipeline
- Real-time human matting using RVM (`mobilenetv3` / `resnet50`)
- Black-avatar rendering mode (privacy-first)
- Corrupted packet tolerance for unstable live streams
- Optional ffmpeg "sanitizer" stage for malformed H264 RTMP sources
- One-command launcher for full black-avatar pipeline

## Repository Layout

- `rtmp_avatar_stream.py`: main Python relay (`input RTMP -> RVM -> output RTMP`)
- `run_rtmp_black_pipeline.sh`: one-file runner (sanitizer + relay + optional viewer)
- `requirements_inference.txt`: runtime dependencies

## Requirements

- macOS/Linux
- Python 3.10+ virtual environment
- FFmpeg installed
- RTMP server endpoint(s)
- RVM model checkpoint (`rvm_mobilenetv3.pth`)

## Setup

```bash
cd /path/to/RVM-rtmp-pipeline
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_inference.txt
```

Download checkpoint:

```bash
curl -L -o rvm_mobilenetv3.pth \
  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth
```

## Quick Start (Single File Runner)

Default (uses local URLs already configured in the script):

```bash
bash run_rtmp_black_pipeline.sh
```

Custom URLs:

```bash
INPUT_RTMP="rtmp://127.0.0.1/live/H1cKHxHYbe" \
OUTPUT_RTMP="rtmp://127.0.0.1/live/H1cKHxHYbe_out" \
bash run_rtmp_black_pipeline.sh
```

## Manual Pipeline

### 1) Start input sanitizer

```bash
ffmpeg -fflags +genpts+discardcorrupt -err_detect ignore_err \
  -use_wallclock_as_timestamps 1 \
  -i "rtmp://127.0.0.1/live/H1cKHxHYbe" \
  -an \
  -c:v libx264 -preset veryfast -tune zerolatency \
  -r 30 -g 60 -keyint_min 60 -sc_threshold 0 \
  -pix_fmt yuv420p \
  -f flv "rtmp://127.0.0.1/live/H1cKHxHYbe_clean"
```

### 2) Run avatar relay

```bash
source .venv/bin/activate
python rtmp_avatar_stream.py \
  --variant mobilenetv3 \
  --checkpoint rvm_mobilenetv3.pth \
  --device mps \
  --input-rtmp "rtmp://127.0.0.1/live/H1cKHxHYbe_clean" \
  --output-rtmp "rtmp://127.0.0.1/live/H1cKHxHYbe_out" \
  --mode black \
  --silhouette-color 0 0 0 \
  --background-color 255 255 255 \
  --input-resize 1280 720 \
  --downsample-ratio 0.25
```

### 3) View output

```bash
ffplay "rtmp://127.0.0.1/live/H1cKHxHYbe_out"
```

## Device Notes

- Apple Silicon Mac: use `--device mps`
- NVIDIA CUDA machine: use `--device cuda`
- CPU fallback: use `--device cpu` (slow)

## Troubleshooting

- `Torch not compiled with CUDA enabled`: use `--device mps` or `--device cpu`.
- `FileNotFoundError: rvm_mobilenetv3.pth`: download checkpoint to repo root.
- `no frame!` / `Invalid data found when processing input`: run the ffmpeg sanitizer stage and use `_clean` stream as input.
- If input reports absurd FPS (for example `1000/1`), script falls back to safe 30 FPS internally.

## Credits

This project builds on the original Robust Video Matting repository and model weights:

- Upstream: https://github.com/PeterL1n/RobustVideoMatting
- Paper: "Robust High-Resolution Video Matting with Temporal Guidance"

## License

GPL-3.0.
