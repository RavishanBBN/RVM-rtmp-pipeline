"""
Real-time RTMP avatar streaming with Robust Video Matting (RVM).

Pipeline:
RTMP input -> RVM alpha matte -> avatar render (black or emoji) -> RTMP output
"""

import argparse
from fractions import Fraction
from typing import Optional, Tuple

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from av.error import InvalidDataError

from model import MattingNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=["mobilenetv3", "resnet50"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--input-rtmp", type=str, required=True)
    parser.add_argument("--output-rtmp", type=str, required=True)
    parser.add_argument("--mode", type=str, default="black", choices=["black", "emoji"])
    parser.add_argument("--emoji-path", type=str, default=None)
    parser.add_argument("--emoji-tile-size", type=int, default=128)
    parser.add_argument("--background-color", type=int, nargs=3, default=[255, 255, 255],
                        help="RGB color used for non-person background, e.g. 255 255 255")
    parser.add_argument("--silhouette-color", type=int, nargs=3, default=[0, 0, 0],
                        help="RGB color used for black avatar mode, e.g. 0 0 0")
    parser.add_argument("--downsample-ratio", type=float, default=None)
    parser.add_argument("--input-resize", type=int, nargs=2, default=None,
                        help="Optional resize in W H. Output stream follows this size.")
    parser.add_argument("--bitrate-mbps", type=float, default=4.0)
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


def tensor_to_video_frame(frame_tensor: torch.Tensor) -> av.VideoFrame:
    frame_u8 = frame_tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return av.VideoFrame.from_ndarray(frame_u8, format="rgb24")


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


def stream_avatar(args):
    if args.mode == "emoji" and not args.emoji_path:
        raise ValueError("--emoji-path is required when --mode emoji")

    device = torch.device(args.device)
    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else torch.float32

    model = MattingNetwork(args.variant).eval().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    if dtype == torch.float16:
        model = model.half()

    input_container = av.open(args.input_rtmp, mode="r")
    input_stream = next(s for s in input_container.streams if s.type == "video")
    stream_rate = resolve_stream_rate(input_stream)
    fps_float = stream_rate.numerator / stream_rate.denominator

    first_decoded = None
    for packet in input_container.demux(input_stream):
        try:
            for frame in packet.decode():
                first_decoded = frame
                break
        except InvalidDataError:
            # Live RTMP streams may contain broken packets; skip and continue.
            continue
        if first_decoded is not None:
            break
    if first_decoded is None:
        raise RuntimeError("No video frame received from input RTMP stream.")

    src0 = first_decoded.to_ndarray(format="rgb24")
    in_h, in_w = src0.shape[:2]
    if args.input_resize is not None:
        out_w, out_h = args.input_resize
    else:
        out_w, out_h = in_w, in_h

    output_container = av.open(args.output_rtmp, mode="w", format="flv")
    output_stream = output_container.add_stream("libx264", rate=stream_rate)
    output_stream.width = out_w
    output_stream.height = out_h
    output_stream.pix_fmt = "yuv420p"
    output_stream.time_base = Fraction(1, max(1, int(round(fps_float))))
    output_stream.bit_rate = int(args.bitrate_mbps * 1_000_000)
    output_stream.codec_context.options = {
        "preset": "veryfast",
        "tune": "zerolatency",
    }

    background_rgb = rgb_triplet_to_tensor(tuple(args.background_color), device, dtype)
    silhouette_rgb = rgb_triplet_to_tensor(tuple(args.silhouette_color), device, dtype)
    emoji_rgba: Optional[torch.Tensor] = None
    if args.mode == "emoji":
        emoji_rgba = load_emoji_rgba(args.emoji_path, device, dtype)

    rec = [None] * 4
    frame_index = 0

    def process_frame(rgb_frame: np.ndarray):
        nonlocal rec, frame_index

        src = torch.from_numpy(rgb_frame).to(device=device)
        src = src.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32).div_(255.0)
        if args.input_resize is not None:
            src = F.interpolate(src, size=(out_h, out_w), mode="bilinear", align_corners=False)
        src = src.to(dtype=dtype)

        ds_ratio = args.downsample_ratio
        if ds_ratio is None:
            ds_ratio = auto_downsample_ratio(src.shape[-2], src.shape[-1])

        with torch.no_grad():
            fgr, pha, *rec = model(src, *rec, ds_ratio)

        if args.mode == "black":
            person = silhouette_rgb.expand_as(fgr)
        else:
            person = build_emoji_fill(
                emoji_rgba=emoji_rgba,
                h=fgr.shape[-2],
                w=fgr.shape[-1],
                tile_size=args.emoji_tile_size,
                fallback_rgb=silhouette_rgb,
            )

        composed = person * pha + background_rgb * (1 - pha)
        out_frame = tensor_to_video_frame(composed[0])
        out_frame.pts = frame_index
        frame_index += 1
        for packet in output_stream.encode(out_frame):
            output_container.mux(packet)

    try:
        process_frame(src0)
        for packet in input_container.demux(input_stream):
            try:
                decoded_frames = packet.decode()
            except InvalidDataError:
                # Keep streaming even if some packets are damaged.
                continue
            for frame in decoded_frames:
                src = frame.to_ndarray(format="rgb24")
                process_frame(src)
    finally:
        for packet in output_stream.encode():
            output_container.mux(packet)
        output_container.close()
        input_container.close()


if __name__ == "__main__":
    stream_avatar(parse_args())
