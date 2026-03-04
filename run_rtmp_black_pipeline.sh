#!/usr/bin/env bash
set -euo pipefail

# Single-file pipeline:
#   RTMP input -> ffmpeg sanitizer -> RVM black avatar relay -> optional ffplay viewer
#
# Usage:
#   bash run_rtmp_black_pipeline.sh
#   INPUT_RTMP="rtmp://127.0.0.1/live/H1cKHxHYbe" bash run_rtmp_black_pipeline.sh
#   INPUT_RTMP="..." OUTPUT_RTMP="rtmp://127.0.0.1/live/H1cKHxHYbe_out" VIEW_OUTPUT=0 bash run_rtmp_black_pipeline.sh

ROOT_DIR="/Users/nbal0029/Desktop/Hattan/RVM/RobustVideoMatting"
VENV_PATH="$ROOT_DIR/.venv"
CHECKPOINT_PATH="$ROOT_DIR/rvm_mobilenetv3.pth"

INPUT_RTMP="${INPUT_RTMP:-rtmp://127.0.0.1/live/H1cKHxHYbe}"
CLEAN_RTMP="${CLEAN_RTMP:-rtmp://127.0.0.1/live/H1cKHxHYbe_clean}"
OUTPUT_RTMP="${OUTPUT_RTMP:-rtmp://127.0.0.1/live/H1cKHxHYbe_out}"

DEVICE="${DEVICE:-mps}"
INPUT_RESIZE_W="${INPUT_RESIZE_W:-1280}"
INPUT_RESIZE_H="${INPUT_RESIZE_H:-720}"
DOWNSAMPLE_RATIO="${DOWNSAMPLE_RATIO:-0.25}"
VIEW_OUTPUT="${VIEW_OUTPUT:-1}"  # 1 to auto-open ffplay, 0 to disable

cleanup() {
  if [[ -n "${SANITIZER_PID:-}" ]] && kill -0 "$SANITIZER_PID" >/dev/null 2>&1; then
    kill "$SANITIZER_PID" || true
  fi
}
trap cleanup EXIT INT TERM

cd "$ROOT_DIR"
source "$VENV_PATH/bin/activate"

python -c "import av, torch, pims; print('av', av.__version__, 'torch', torch.__version__)"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  curl -L -o "$CHECKPOINT_PATH" \
    https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth
fi

echo "Starting sanitizer stream: $INPUT_RTMP -> $CLEAN_RTMP"
ffmpeg -hide_banner -loglevel warning \
  -fflags +genpts+discardcorrupt -err_detect ignore_err \
  -use_wallclock_as_timestamps 1 \
  -i "$INPUT_RTMP" \
  -an \
  -c:v libx264 -preset veryfast -tune zerolatency \
  -r 30 -g 60 -keyint_min 60 -sc_threshold 0 \
  -pix_fmt yuv420p \
  -f flv "$CLEAN_RTMP" &
SANITIZER_PID=$!

sleep 2

echo "Probing sanitized stream..."
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,avg_frame_rate,r_frame_rate \
  -of default=nw=1 "$CLEAN_RTMP" || true

if [[ "$VIEW_OUTPUT" == "1" ]]; then
  if command -v ffplay >/dev/null 2>&1; then
    echo "Opening output viewer at $OUTPUT_RTMP"
    ffplay -fflags nobuffer -flags low_delay "$OUTPUT_RTMP" >/dev/null 2>&1 &
  else
    echo "ffplay not found; skipping viewer."
  fi
fi

echo "Starting RVM black avatar relay: $CLEAN_RTMP -> $OUTPUT_RTMP"
python rtmp_avatar_stream.py \
  --variant mobilenetv3 \
  --checkpoint "$CHECKPOINT_PATH" \
  --device "$DEVICE" \
  --input-rtmp "$CLEAN_RTMP" \
  --output-rtmp "$OUTPUT_RTMP" \
  --mode black \
  --silhouette-color 0 0 0 \
  --background-color 255 255 255 \
  --input-resize "$INPUT_RESIZE_W" "$INPUT_RESIZE_H" \
  --downsample-ratio "$DOWNSAMPLE_RATIO"
