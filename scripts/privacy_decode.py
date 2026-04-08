#!/usr/bin/env python3
"""
Privacy-Preserving H.264 Decoder

Decodes an H.264 stream with privacy metadata.  If a valid passkey is
provided, the original video is reconstructed.  Without a passkey, the
sensitive regions identified by the mask are obfuscated.

Usage:
    # Decode WITHOUT passkey (privacy mode - faces are blurred):
    python privacy_decode.py --input privacy_encoded.mp4 --output censored.mp4

    # Decode WITH passkey (authorized mode - original video):
    python privacy_decode.py --input privacy_encoded.mp4 --output original.mp4 \
        --passkey <hex_key>

    # Decode with mask sidecar file:
    python privacy_decode.py --input video.mp4 --mask video.mp4.mask \
        --output censored.mp4 --width 1920 --height 1080
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np


OBFUSCATION_TYPES = {
    "pixelate": 0,
    "blur": 1,
    "black": 2,
    "invert": 3,
}


def get_ffmpeg_path():
    """Return path to the locally-built ffmpeg, falling back to system."""
    local = os.path.join(os.path.dirname(__file__), "..", "ffmpeg")
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local
    return "ffmpeg"


def probe_video(video_path):
    """Return (width, height, fps) of the video."""
    ffmpeg = get_ffmpeg_path()
    result = subprocess.run(
        [ffmpeg, "-i", video_path],
        capture_output=True, text=True
    )
    width = height = None
    fps = 30.0
    for line in (result.stdout + result.stderr).split("\n"):
        if "Video:" in line:
            m = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if m:
                width, height = int(m.group(1)), int(m.group(2))
            m_fps = re.search(r"([\d.]+)\s*fps", line)
            if m_fps:
                fps = float(m_fps.group(1))
            break
    return width, height, fps


def apply_pixelation(frame, mask, block_size=16):
    """Apply pixelation to masked regions."""
    h, w = frame.shape[:2]
    result = frame.copy()

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y1 = min(y + block_size, h)
            x1 = min(x + block_size, w)

            # Check if this block is masked
            block_mask = mask[y:y1, x:x1]
            if np.mean(block_mask) > 127:
                # Replace with average color
                block = frame[y:y1, x:x1]
                avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
                result[y:y1, x:x1] = avg_color

    return result


def apply_blur(frame, mask, ksize=31):
    """Apply Gaussian blur to masked regions."""
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    # Expand mask to 3 channels
    mask_3ch = np.stack([mask, mask, mask], axis=-1) if len(frame.shape) == 3 else mask
    mask_float = (mask_3ch > 127).astype(np.float32)
    result = (frame * (1 - mask_float) + blurred * mask_float).astype(np.uint8)
    return result


def apply_black(frame, mask):
    """Apply black fill to masked regions."""
    result = frame.copy()
    if len(frame.shape) == 3:
        result[mask > 127] = [0, 0, 0]
    else:
        result[mask > 127] = 0
    return result


def apply_invert(frame, mask):
    """Apply color inversion to masked regions."""
    result = frame.copy()
    if len(frame.shape) == 3:
        mask_3d = np.stack([mask > 127] * 3, axis=-1)
        result[mask_3d] = 255 - result[mask_3d]
    else:
        result[mask > 127] = 255 - result[mask > 127]
    return result


def decode_with_privacy(input_video, output_video, mask_file=None,
                        passkey=None, obfuscation="pixelate",
                        width=None, height=None):
    """
    Decode a privacy-encoded H.264 stream.

    If passkey is provided, outputs the original video.
    Otherwise, applies obfuscation to masked regions.
    """
    ffmpeg = get_ffmpeg_path()

    # Try to use the privacy_mask filter first
    if passkey:
        # Authorized mode: use privacy_mask filter with passkey
        filter_opts = f"privacy_mask=passkey={passkey}"
        cmd = [
            ffmpeg, "-y",
            "-i", input_video,
            "-vf", filter_opts,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            output_video,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Authorized decode complete: {output_video}")
            return output_video

    # Fallback: use OpenCV-based decoding with mask sidecar
    det_width, det_height, fps = probe_video(input_video)
    if width is None:
        width = det_width
    if height is None:
        height = det_height

    if width is None or height is None:
        raise RuntimeError("Cannot determine video dimensions")

    # If passkey is provided, just copy the video through
    if passkey:
        cmd = [
            ffmpeg, "-y",
            "-i", input_video,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            output_video,
        ]
        subprocess.run(cmd, check=True)
        print(f"Authorized decode (passthrough): {output_video}")
        return output_video

    # Check for mask sidecar file
    if mask_file is None:
        mask_file = input_video + ".mask"

    if not os.path.isfile(mask_file):
        print(f"No mask file found at {mask_file}. "
              "Outputting original video (no obfuscation).")
        cmd = [
            ffmpeg, "-y",
            "-i", input_video,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            output_video,
        ]
        subprocess.run(cmd, check=True)
        return output_video

    print(f"Privacy mode: applying {obfuscation} to masked regions")
    print(f"Mask file: {mask_file}, Resolution: {width}x{height}")

    # Read video frames and apply obfuscation
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_size = width * height
    mask_fp = open(mask_file, "rb")

    frame_idx = 0
    last_mask = np.zeros((height, width), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        # Read mask frame
        mask_data = mask_fp.read(frame_size)
        if len(mask_data) == frame_size:
            last_mask = np.frombuffer(mask_data, dtype=np.uint8).reshape(height, width)

        # Apply obfuscation
        if obfuscation == "pixelate":
            frame = apply_pixelation(frame, last_mask, block_size=16)
        elif obfuscation == "blur":
            frame = apply_blur(frame, last_mask, ksize=31)
        elif obfuscation == "black":
            frame = apply_black(frame, last_mask)
        elif obfuscation == "invert":
            frame = apply_invert(frame, last_mask)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx} frames...")

    cap.release()
    out.release()
    mask_fp.close()

    print(f"Privacy-compliant decode complete: {output_video} ({frame_idx} frames)")
    return output_video


def main():
    parser = argparse.ArgumentParser(
        description="Privacy-preserving H.264 decoder"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input H.264/MP4 file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output video file")
    parser.add_argument("--mask", "-m", default=None,
                        help="Mask sidecar file (raw grayscale frames)")
    parser.add_argument("--passkey", "-k", default=None,
                        help="Passkey for authorized decoding (bypasses obfuscation)")
    parser.add_argument("--obfuscation", choices=OBFUSCATION_TYPES.keys(),
                        default="pixelate",
                        help="Obfuscation type for privacy mode")
    parser.add_argument("--width", type=int, default=None,
                        help="Video width (auto-detected if omitted)")
    parser.add_argument("--height", type=int, default=None,
                        help="Video height (auto-detected if omitted)")

    args = parser.parse_args()

    decode_with_privacy(
        args.input, args.output,
        mask_file=args.mask,
        passkey=args.passkey,
        obfuscation=args.obfuscation,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
