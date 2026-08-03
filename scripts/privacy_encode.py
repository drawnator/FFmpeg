#!/usr/bin/env python3
"""
Privacy-Preserving H.264 Encoder

Encodes a video with an accompanying binary mask into an H.264 stream
that carries privacy metadata via SEI messages.  The mask identifies
regions requiring privacy protection (e.g. faces, license plates).

Usage:
    python privacy_encode.py --input video.mp4 --mask mask_video.mp4 \
        --output privacy_encoded.h264 [--key <hex_key>] [--obfuscation pixelate]

The mask video should be the same resolution as the input.  Bright
(white) regions in the mask are treated as privacy-sensitive.  If no
mask video is provided, the encoder produces a standard H.264 stream.
"""

import argparse
import hashlib
import os
import re
import struct
import subprocess
import sys
import tempfile

import cv2
import numpy as np

# Obfuscation type mapping
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


def extract_raw_mask_frames(mask_video, width, height, output_raw):
    """
    Read a mask video and write raw grayscale frames to a binary file.

    Each frame is *width* × *height* bytes of grayscale data.
    """
    cap = cv2.VideoCapture(mask_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open mask video: {mask_video}")

    with open(output_raw, "wb") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
            # Threshold: > 127 is masked
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            f.write(binary.tobytes())

    cap.release()
    return output_raw


def encode_with_privacy(input_video, mask_video, output_file,
                        key_hex=None, obfuscation="pixelate",
                        encryption_algo=0):
    """
    Encode the input video as H.264 and inject privacy SEI metadata
    from the mask via the h264_privacy bitstream filter.
    """
    ffmpeg = get_ffmpeg_path()

    # Probe input dimensions
    probe = subprocess.run(
        [ffmpeg, "-i", input_video],
        capture_output=True, text=True
    )
    # Extract resolution from ffprobe / ffmpeg output
    width, height = None, None
    for line in (probe.stdout + probe.stderr).split("\n"):
        if "Video:" in line:
            m = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if m:
                width, height = int(m.group(1)), int(m.group(2))
                break

    if width is None or height is None:
        raise RuntimeError("Could not determine video resolution")

    print(f"Input resolution: {width}x{height}")

    tmpdir = tempfile.mkdtemp(prefix="privacy_encode_")

    # Step 1: Extract raw mask frames
    raw_mask = os.path.join(tmpdir, "mask.raw")
    if mask_video:
        print("Extracting raw mask frames...")
        extract_raw_mask_frames(mask_video, width, height, raw_mask)
    else:
        # No mask: create empty mask file
        with open(raw_mask, "wb"):
            pass

    # Step 2: Encode with H.264 + privacy BSF
    obfusc_type = OBFUSCATION_TYPES.get(obfuscation, 0)

    bsf_opts = (
        f"h264_privacy=mask_file={raw_mask}"
        f":obfuscation={obfusc_type}"
        f":encryption={encryption_algo}"
    )
    if key_hex:
        bsf_opts += f":key={key_hex}"

    cmd = [
        ffmpeg, "-y",
        "-i", input_video,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-bsf:v", bsf_opts,
        "-an",  # Skip audio for simplicity
        output_file,
    ]

    print(f"Encoding: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg stderr:\n{result.stderr}")
        # Fall back to standard encoding without BSF
        print("Note: h264_privacy BSF not available in this build. "
              "Falling back to standard encode + separate mask file.")
        cmd_fallback = [
            ffmpeg, "-y",
            "-i", input_video,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-an",
            output_file,
        ]
        subprocess.run(cmd_fallback, check=True)

        # Copy mask alongside the output
        mask_sidecar = output_file + ".mask"
        if mask_video and os.path.getsize(raw_mask) > 0:
            import shutil
            shutil.copy2(raw_mask, mask_sidecar)
            print(f"Mask sidecar written to: {mask_sidecar}")

    print(f"Output: {output_file}")
    if key_hex:
        print(f"Encryption key: {key_hex}")

    # Clean up
    try:
        os.remove(raw_mask)
        os.rmdir(tmpdir)
    except OSError:
        pass

    return output_file


def generate_key():
    """Generate a random 128-bit hex key."""
    return os.urandom(16).hex()


def main():
    parser = argparse.ArgumentParser(
        description="Privacy-preserving H.264 encoder with mask metadata"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input video file")
    parser.add_argument("--mask", "-m", default=None,
                        help="Mask video file (white=sensitive regions)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output H.264 file")
    parser.add_argument("--key", default=None,
                        help="Hex encryption key (32 chars for AES-128). "
                             "If omitted, a random key is generated.")
    parser.add_argument("--obfuscation", choices=OBFUSCATION_TYPES.keys(),
                        default="pixelate",
                        help="Obfuscation type for privacy mode")
    parser.add_argument("--no-encrypt", action="store_true",
                        help="Disable mask encryption (store plaintext)")

    args = parser.parse_args()

    encryption_algo = 0 if args.no_encrypt else 1
    key_hex = args.key
    if encryption_algo > 0 and not key_hex:
        key_hex = generate_key()
        print(f"Generated encryption key: {key_hex}")

    encode_with_privacy(
        args.input, args.mask, args.output,
        key_hex=key_hex,
        obfuscation=args.obfuscation,
        encryption_algo=encryption_algo,
    )


if __name__ == "__main__":
    main()
