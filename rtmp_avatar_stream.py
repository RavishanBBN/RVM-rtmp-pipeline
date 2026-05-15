"""
Real-time RTMP avatar streaming with Robust Video Matting (RVM).

Pipeline:
RTMP input -> RVM alpha matte -> avatar render -> RTMP output
"""

import argparse
import queue
import subprocess
import threading
import time
import traceback
from fractions import Fraction
from typing import Optional, Tuple

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from av.error import FFmpegError, InvalidDataError

from model import MattingNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=["mobilenetv3", "resnet50"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--input-rtmp", type=str, required=True)
    parser.add_argument("--output-rtmp", type=str, required=True)
    parser.add_argument("--mode", type=str, default="black", choices=["black", "foreground", "background", "emoji", "passthrough"])
    parser.add_argument("--emoji-path", type=str, default=None)
    parser.add_argument("--emoji-tile-size", type=int, default=128)
    parser.add_argument("--background-color", type=int, nargs=3, default=[255, 255, 255],
                        help="RGB color used for non-person background, e.g. 255 255 255")
    parser.add_argument("--silhouette-color", type=int, nargs=3, default=[0, 0, 0],
                        help="RGB color used for black avatar mode, e.g. 0 0 0")
    parser.add_argument(
        "--hard-mask-threshold",
        type=float,
        default=None,
        help="Optional threshold in [0,1]. If set, alpha is binarized for strict foreground/background separation.",
    )
    parser.add_argument("--downsample-ratio", type=float, default=None)
    parser.add_argument("--input-resize", type=int, nargs=2, default=None,
                        help="Optional resize in W H. Output stream follows this size.")
    parser.add_argument("--input-fps", type=float, default=10.0,
                        help="FPS requested from the ffmpeg input pipe before optional frame skipping.")
    parser.add_argument("--frame-timeout-seconds", type=float, default=8.0,
                        help="Reconnect if no decoded input frame is received within this many seconds.")
    parser.add_argument("--input-queue-size", type=int, default=1,
                        help="Number of latest decoded frames to buffer. Keep this low for live latency.")
    parser.add_argument("--bitrate-mbps", type=float, default=4.0)
    parser.add_argument("--process-every-nth-frame", type=int, default=1,
                        help="Only run inference on every Nth decoded frame to keep live latency bounded.")
    parser.add_argument("--reconnect-delay-seconds", type=float, default=2.0,
                        help="Seconds to wait before reconnecting after a live stream error.")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def auto_downsample_ratio(h: int, w: int) -> float:
    return min(512 / max(h, w), 1.0)


def rgb_triplet_to_tensor(rgb: Tuple[int, int, int], device: torch.device, dtype: torch.dtype):
    return torch.tensor(rgb, device=device, dtype=dtype).view(1, 3, 1, 1).div(255.0)


def load_emoji_rgba(path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    with Image.open(path) as img:
        rgba = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    rgba = torch.from_numpy(rgba).permute(2, 0, 1).to(device=device, dtype=dtype)  # [4, H, W]
    return rgba


def build_emoji_fill(
    emoji_rgba: torch.Tensor,
    h: int,
    w: int,
    tile_size: int,
    fallback_rgb: torch.Tensor
) -> torch.Tensor:
    tile = F.interpolate(
        emoji_rgba.unsqueeze(0),
        size=(tile_size, tile_size),
        mode="bilinear",
        align_corners=False,
    )[0]  # [4, T, T]
    reps_h = (h + tile_size - 1) // tile_size
    reps_w = (w + tile_size - 1) // tile_size
    tiled = tile.repeat(1, reps_h, reps_w)[:, :h, :w]  # [4, H, W]
    rgb = tiled[:3].unsqueeze(0)  # [1, 3, H, W]
    a = tiled[3:4].unsqueeze(0)   # [1, 1, H, W]
    return rgb * a + fallback_rgb * (1 - a)


def tensor_to_rgb_ndarray(frame_tensor: torch.Tensor) -> np.ndarray:
    frame_u8 = frame_tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return np.ascontiguousarray(frame_u8)


class FfmpegRtmpWriter:
    def __init__(self, url: str, width: int, height: int, fps: Fraction, bitrate_mbps: float):
        fps_float = fps.numerator / fps.denominator
        gop = max(1, int(round(fps_float)))
        bitrate = f"{bitrate_mbps:g}M"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-nostdin",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s:v", f"{width}x{height}",
            "-r", f"{fps.numerator}/{fps.denominator}",
            "-i", "pipe:0",
            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-pix_fmt", "yuv420p",
            "-g", str(gop),
            "-keyint_min", str(gop),
            "-sc_threshold", "0",
            "-bf", "0",
            "-b:v", bitrate,
            "-f", "flv",
            url,
        ]
        print("[relay] ffmpeg output command: " + " ".join(command), flush=True)
        self.process = subprocess.Popen(
            command,
            bufsize=0,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )

    def write(self, rgb_frame: np.ndarray):
        if self.process.poll() is not None:
            raise RuntimeError(f"ffmpeg output exited with code {self.process.returncode}")
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg output stdin is closed")
        self.process.stdin.write(rgb_frame.tobytes())
        self.process.stdin.flush()

    def close(self):
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


class FfmpegRawVideoReader:
    def __init__(self, url: str, width: int, height: int, fps: float, queue_size: int):
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=max(1, queue_size))
        self.error: Optional[BaseException] = None
        self.closed = False
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-nostdin",
            "-fflags", "+genpts+discardcorrupt",
            "-flags", "low_delay",
            "-err_detect", "ignore_err",
            "-rtmp_live", "live",
            "-i", url,
            "-an",
            "-vf", f"scale={width}:{height},fps={fps:g}",
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "pipe:1",
        ]
        print("[relay] ffmpeg input command: " + " ".join(command), flush=True)
        self.process = subprocess.Popen(
            command,
            bufsize=0,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
        )
        self.thread = threading.Thread(target=self._read_loop, name="ffmpeg-input-reader", daemon=True)
        self.thread.start()

    def _read_exact(self) -> bytes:
        if self.process.stdout is None:
            raise RuntimeError("ffmpeg input stdout is closed")

        buffer = bytearray()
        while len(buffer) < self.frame_size:
            chunk = self.process.stdout.read(self.frame_size - len(buffer))
            if not chunk:
                raise RuntimeError("ffmpeg input ended before a complete frame was read")
            buffer.extend(chunk)
        return bytes(buffer)

    def _read_loop(self):
        try:
            while not self.closed:
                if self.process.poll() is not None:
                    raise RuntimeError(f"ffmpeg input exited with code {self.process.returncode}")
                data = self._read_exact()
                frame = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))
                frame = np.ascontiguousarray(frame)
                if self.frames.full():
                    try:
                        self.frames.get_nowait()
                    except queue.Empty:
                        pass
                self.frames.put_nowait(frame)
        except BaseException as exc:
            if not self.closed:
                self.error = exc

    def read(self, timeout_seconds: float) -> np.ndarray:
        try:
            return self.frames.get(timeout=max(0.1, timeout_seconds))
        except queue.Empty as exc:
            if self.error is not None:
                raise RuntimeError(f"ffmpeg input reader stopped: {self.error}") from self.error
            if self.process.poll() is not None:
                raise RuntimeError(f"ffmpeg input exited with code {self.process.returncode}") from exc
            raise RuntimeError(f"timed out waiting for input frame after {timeout_seconds:.1f}s") from exc

    def close(self):
        self.closed = True
        if self.process.stdout is not None:
            try:
                self.process.stdout.close()
            except OSError:
                pass
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.thread.join(timeout=2)


def resolve_stream_rate(input_stream) -> Fraction:
    # PyAV expects rate as int/Fraction for add_stream.
    candidates = [
        getattr(input_stream, "average_rate", None),
        getattr(input_stream, "base_rate", None),
        getattr(input_stream, "guessed_rate", None),
    ]
    for rate in candidates:
        if rate is None:
            continue
        try:
            if rate.numerator > 0 and rate.denominator > 0:
                resolved = Fraction(rate.numerator, rate.denominator)
                fps = resolved.numerator / resolved.denominator
                if 1 <= fps <= 120:
                    return resolved
        except AttributeError:
            if isinstance(rate, (int, float)) and rate > 0:
                fps = float(rate)
                if 1 <= fps <= 120:
                    return Fraction(int(round(fps)), 1)
    # Some RTMP sources expose bogus rates like 1000/1. Fall back to sane default.
    return Fraction(30, 1)


def open_input_container(url: str, max_retries=10, timeout=10.0):
    for attempt in range(max_retries):
        try:
            print(f"Connecting to input stream (Attempt {attempt+1}/{max_retries}): {url}")
            container = av.open(
                url,
                mode="r",
                timeout=timeout,
                options={
                    "rtmp_live": "live",
                    "fflags": "genpts+discardcorrupt",
                    "err_detect": "ignore_err",
                    "use_wallclock_as_timestamps": "1",
                    "rw_timeout": str(int(timeout * 1000000)),
                },
            )
            return container
        except FFmpegError as e:
            print(f"Failed to connect: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise


def open_output_container(url: str, timeout=10.0):
    return av.open(
        url,
        mode="w",
        format="flv",
        timeout=timeout,
        options={
            "rtmp_live": "live",
            "rw_timeout": str(int(timeout * 1000000)),
        },
    )


def stream_avatar_session(args, model, device, dtype, background_rgb, silhouette_rgb, emoji_rgba):
    if args.mode == "emoji" and not args.emoji_path:
        raise ValueError("--emoji-path is required when --mode emoji")
    if args.hard_mask_threshold is not None and not (0.0 <= args.hard_mask_threshold <= 1.0):
        raise ValueError("--hard-mask-threshold must be within [0, 1]")
    if args.input_resize is None:
        raise ValueError("--input-resize W H is required for the live ffmpeg pipe path")
    if args.input_fps <= 0:
        raise ValueError("--input-fps must be greater than 0")
    if args.frame_timeout_seconds <= 0:
        raise ValueError("--frame-timeout-seconds must be greater than 0")

    out_w, out_h = args.input_resize
    print(f"[relay] opening ffmpeg input: {args.input_rtmp}", flush=True)
    input_reader = FfmpegRawVideoReader(
        args.input_rtmp,
        out_w,
        out_h,
        args.input_fps,
        args.input_queue_size,
    )

    output_rate = Fraction(
        max(1, int(round(args.input_fps * 1000))),
        1000 * max(1, args.process_every_nth_frame),
    )
    print(f"[relay] opening ffmpeg output: {args.output_rtmp}", flush=True)
    output_writer = FfmpegRtmpWriter(args.output_rtmp, out_w, out_h, output_rate, args.bitrate_mbps)
    print(f"[relay] streams configured at {out_w}x{out_h}, output ~{output_rate.numerator / output_rate.denominator:.2f} fps", flush=True)

    rec = [None] * 4
    input_frame_count = 0
    first_packet_muxed = False

    def process_frame(rgb_frame: np.ndarray):
        nonlocal rec, first_packet_muxed

        src = torch.from_numpy(rgb_frame).to(device=device)
        src = src.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32).div_(255.0)
        if args.input_resize is not None:
            src = F.interpolate(src, size=(out_h, out_w), mode="bilinear", align_corners=False)
        src = src.to(dtype=dtype)

        ds_ratio = args.downsample_ratio
        if ds_ratio is None:
            ds_ratio = auto_downsample_ratio(src.shape[-2], src.shape[-1])

        if args.mode == "passthrough":
            composed = src
        else:
            with torch.no_grad():
                fgr, pha, *rec = model(src, *rec, ds_ratio)
            if args.hard_mask_threshold is not None:
                pha = (pha >= args.hard_mask_threshold).to(dtype=dtype)

            if args.mode == "foreground":
                # Real person, replaced background.
                composed = fgr * pha + background_rgb * (1 - pha)
            elif args.mode == "background":
                # Replaced person, real background.
                silhouette = silhouette_rgb.expand_as(fgr)
                composed = silhouette * pha + src * (1 - pha)
            elif args.mode == "black":
                person = silhouette_rgb.expand_as(fgr)
                composed = person * pha + background_rgb * (1 - pha)
            else:
                person = build_emoji_fill(
                    emoji_rgba=emoji_rgba,
                    h=fgr.shape[-2],
                    w=fgr.shape[-1],
                    tile_size=args.emoji_tile_size,
                    fallback_rgb=silhouette_rgb,
                )
                composed = person * pha + background_rgb * (1 - pha)

        out_frame = tensor_to_rgb_ndarray(composed[0])
        try:
            output_writer.write(out_frame)
            if not first_packet_muxed:
                first_packet_muxed = True
                print("[relay] first output frame written", flush=True)
        except (BrokenPipeError, OSError, RuntimeError) as exc:
            raise RuntimeError(f"output ffmpeg write failed: {exc}") from exc

    try:
        while True:
            src = input_reader.read(args.frame_timeout_seconds)
            input_frame_count += 1
            if input_frame_count % max(1, args.process_every_nth_frame) != 0:
                continue
            process_frame(src)
    finally:
        try:
            output_writer.close()
        finally:
            input_reader.close()


def stream_avatar(args):
    if args.mode == "emoji" and not args.emoji_path:
        raise ValueError("--emoji-path is required when --mode emoji")
    if args.hard_mask_threshold is not None and not (0.0 <= args.hard_mask_threshold <= 1.0):
        raise ValueError("--hard-mask-threshold must be within [0, 1]")
    if args.process_every_nth_frame < 1:
        raise ValueError("--process-every-nth-frame must be >= 1")

    if args.device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = args.device

    device = torch.device(resolved_device)
    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else torch.float32
    print(f"[relay] using device={device.type} dtype={dtype}", flush=True)

    model = None
    if args.mode != "passthrough":
        print(f"[relay] loading model variant={args.variant} checkpoint={args.checkpoint}", flush=True)
        model = MattingNetwork(args.variant).eval().to(device)
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        if dtype == torch.float16:
            model = model.half()
        print("[relay] model loaded", flush=True)

    background_rgb = rgb_triplet_to_tensor(tuple(args.background_color), device, dtype)
    silhouette_rgb = rgb_triplet_to_tensor(tuple(args.silhouette_color), device, dtype)
    emoji_rgba: Optional[torch.Tensor] = None
    if args.mode == "emoji":
        emoji_rgba = load_emoji_rgba(args.emoji_path, device, dtype)

    while True:
        try:
            print(f"[relay] connecting input={args.input_rtmp} output={args.output_rtmp}", flush=True)
            stream_avatar_session(args, model, device, dtype, background_rgb, silhouette_rgb, emoji_rgba)
            print("[relay] stream ended, reconnecting...", flush=True)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[relay] stream error: {exc}. reconnecting in {args.reconnect_delay_seconds:.1f}s", flush=True)
            print(traceback.format_exc(), flush=True)
        time.sleep(max(0.25, args.reconnect_delay_seconds))


if __name__ == "__main__":
    stream_avatar(parse_args())
