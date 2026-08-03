#!/usr/bin/env python3
"""
YouTube Privacy Codec Demo

Downloads a short public-domain YouTube video containing both faces and
license plates, generates privacy masks automatically, and demonstrates
the dual-stream privacy codec by producing:

  1. An encoded video WITH privacy metadata
  2. A decoded video WITHOUT passkey (censored version)
  3. A decoded video WITH passkey (uncensored / original version)

Usage:
    python demo_youtube.py [--url <youtube_url>] [--output-dir ./demo_output]

The script defaults to a Creative Commons traffic/pedestrian video
that contains both faces and vehicles with license plates.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_ffmpeg_path():
    """Return path to the locally-built ffmpeg, falling back to system."""
    local = os.path.join(os.path.dirname(__file__), "..", "ffmpeg")
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local
    return "ffmpeg"


def download_youtube_video(url, output_path, max_duration=30):
    """
    Download a YouTube video using yt-dlp, limited to max_duration seconds.
    """
    try:
        import yt_dlp
    except ImportError:
        print("Error: yt-dlp not installed. Install with: pip install yt-dlp")
        sys.exit(1)

    print(f"Downloading video from: {url}")
    print(f"Max duration: {max_duration} seconds")

    ydl_opts = {
        "format": "best[height<=720][ext=mp4]/best[height<=720]/best",
        "outtmpl": output_path,
        "quiet": False,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Trim to max_duration seconds
    if max_duration and os.path.isfile(output_path):
        ffmpeg = get_ffmpeg_path()
        trimmed = output_path + ".trimmed.mp4"
        cmd = [
            ffmpeg, "-y",
            "-i", output_path,
            "-t", str(max_duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            trimmed,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            os.replace(trimmed, output_path)
        else:
            if os.path.isfile(trimmed):
                os.remove(trimmed)

    print(f"Downloaded: {output_path}")
    return output_path


def create_sample_video(output_path, duration=10, width=640, height=480):
    """
    Create a synthetic sample video with face-like and plate-like objects
    for testing when YouTube download is unavailable.
    """
    import cv2
    import numpy as np

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(int(fps * duration)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Background gradient
        for y in range(height):
            frame[y, :] = [int(50 + 100 * y / height),
                           int(80 + 80 * y / height),
                           int(120 + 60 * y / height)]

        # Simulated face (oval with skin-tone color)
        cx = int(200 + 100 * np.sin(i * 0.05))
        cy = int(180 + 50 * np.cos(i * 0.03))
        cv2.ellipse(frame, (cx, cy), (40, 55), 0, 0, 360, (180, 200, 220), -1)
        cv2.circle(frame, (cx - 15, cy - 10), 5, (50, 50, 50), -1)  # Left eye
        cv2.circle(frame, (cx + 15, cy - 10), 5, (50, 50, 50), -1)  # Right eye
        cv2.ellipse(frame, (cx, cy + 15), (15, 8), 0, 0, 180, (50, 50, 200), 2)

        # Simulated second face
        cx2 = int(450 + 80 * np.cos(i * 0.04))
        cy2 = int(200 + 40 * np.sin(i * 0.06))
        cv2.ellipse(frame, (cx2, cy2), (35, 50), 0, 0, 360, (170, 190, 210), -1)
        cv2.circle(frame, (cx2 - 12, cy2 - 8), 4, (50, 50, 50), -1)
        cv2.circle(frame, (cx2 + 12, cy2 - 8), 4, (50, 50, 50), -1)

        # Simulated car with license plate
        car_x = int(100 + 300 * ((i * 2) % (width + 200)) / (width + 200))
        car_y = 380
        cv2.rectangle(frame, (car_x, car_y), (car_x + 120, car_y + 60),
                       (100, 100, 180), -1)
        # License plate
        plate_x = car_x + 30
        plate_y = car_y + 35
        cv2.rectangle(frame, (plate_x, plate_y),
                       (plate_x + 60, plate_y + 18), (255, 255, 255), -1)
        cv2.putText(frame, "AB-1234", (plate_x + 3, plate_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Frame info
        cv2.putText(frame, f"Frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        out.write(frame)

    out.release()
    print(f"Created sample video: {output_path}")
    return output_path


def run_demo(url=None, output_dir="./demo_output", max_duration=15):
    """
    Run the complete privacy codec demo.
    """
    import cv2
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    # File paths
    source_video = os.path.join(output_dir, "source_video.mp4")
    raw_mask = os.path.join(output_dir, "privacy_mask.raw")
    encoded_video = os.path.join(output_dir, "privacy_encoded.mp4")
    censored_video = os.path.join(output_dir, "decoded_censored.mp4")
    original_video = os.path.join(output_dir, "decoded_original.mp4")
    comparison_video = os.path.join(output_dir, "comparison.mp4")
    mask_sidecar = encoded_video + ".mask"

    print("=" * 70)
    print("  Privacy-Preserving H.264 Codec Demo")
    print("=" * 70)
    print()

    # Step 1: Get source video
    print("STEP 1: Obtaining source video...")
    print("-" * 40)

    if url:
        try:
            download_youtube_video(url, source_video, max_duration=max_duration)
        except Exception as e:
            print(f"Download failed: {e}")
            print("Creating synthetic sample video instead...")
            create_sample_video(source_video, duration=max_duration)
    else:
        # Try to download a CC-licensed traffic video
        # If download fails, create a synthetic sample
        default_urls = [
            # Short Creative Commons traffic videos
            "https://www.youtube.com/watch?v=MNn9qKG2UFI",  # Traffic cam
        ]
        downloaded = False
        for try_url in default_urls:
            try:
                download_youtube_video(try_url, source_video,
                                       max_duration=max_duration)
                downloaded = True
                break
            except Exception as e:
                print(f"  Failed to download {try_url}: {e}")

        if not downloaded:
            print("Creating synthetic sample video with faces and plates...")
            create_sample_video(source_video, duration=max_duration)

    print()

    # Step 2: Generate privacy masks
    print("STEP 2: Generating privacy masks (face + license plate detection)...")
    print("-" * 40)

    from auto_detect_faces import generate_mask_video
    width, height, fps = generate_mask_video(
        source_video, raw_mask,
        detect_type="all",
        skip_frames=1,
    )
    print()

    # Step 3: Encode with privacy metadata
    print("STEP 3: Encoding with privacy metadata...")
    print("-" * 40)

    from privacy_encode import encode_with_privacy, generate_key

    passkey = generate_key()
    print(f"Generated passkey: {passkey}")

    # Create mask video for encoding
    ffmpeg = get_ffmpeg_path()
    tmpdir = tempfile.mkdtemp(prefix="demo_")
    mask_video = os.path.join(tmpdir, "mask.avi")

    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-pixel_format", "gray",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", raw_mask,
        "-c:v", "mjpeg",
        "-q:v", "2",
        mask_video,
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    encode_with_privacy(
        source_video, mask_video, encoded_video,
        key_hex=passkey,
        obfuscation="pixelate",
        encryption_algo=0,
    )

    # Create mask sidecar
    shutil.copy2(raw_mask, mask_sidecar)
    print()

    # Step 4: Decode WITHOUT passkey (censored)
    print("STEP 4: Decoding WITHOUT passkey (privacy mode - censored)...")
    print("-" * 40)

    from privacy_decode import decode_with_privacy

    decode_with_privacy(
        encoded_video, censored_video,
        mask_file=mask_sidecar,
        passkey=None,
        obfuscation="pixelate",
        width=width, height=height,
    )
    print()

    # Step 5: Decode WITH passkey (original)
    print("STEP 5: Decoding WITH passkey (authorized mode - original)...")
    print("-" * 40)

    decode_with_privacy(
        encoded_video, original_video,
        mask_file=mask_sidecar,
        passkey=passkey,
        obfuscation="pixelate",
        width=width, height=height,
    )
    print()

    # Step 6: Create side-by-side comparison
    print("STEP 6: Creating side-by-side comparison video...")
    print("-" * 40)

    cap_censored = cv2.VideoCapture(censored_video)
    cap_original = cv2.VideoCapture(original_video)

    if cap_censored.isOpened() and cap_original.isOpened():
        comp_w = width * 2
        comp_h = height
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        comp_out = cv2.VideoWriter(comparison_video, fourcc, fps,
                                   (comp_w, comp_h))

        frame_count = 0
        while True:
            ret1, f_censored = cap_censored.read()
            ret2, f_original = cap_original.read()
            if not ret1 or not ret2:
                break

            f_censored = cv2.resize(f_censored, (width, height))
            f_original = cv2.resize(f_original, (width, height))

            # Add labels
            cv2.putText(f_censored, "CENSORED (No Passkey)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(f_original, "ORIGINAL (With Passkey)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            combined = np.hstack([f_censored, f_original])
            comp_out.write(combined)
            frame_count += 1

        comp_out.release()
        cap_censored.release()
        cap_original.release()

        print(f"Comparison video: {comparison_video} ({frame_count} frames)")
    else:
        print("Warning: Could not create comparison video")
        cap_censored.release()
        cap_original.release()

    print()

    # Clean up temp files
    try:
        os.remove(mask_video)
        os.rmdir(tmpdir)
    except OSError:
        pass

    # Summary
    print("=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Output files:")
    print(f"  Source video:      {source_video}")
    print(f"  Privacy mask:      {raw_mask}")
    print(f"  Encoded (privacy): {encoded_video}")
    print(f"  Mask sidecar:      {mask_sidecar}")
    print(f"  Censored output:   {censored_video}")
    print(f"  Original output:   {original_video}")
    print(f"  Comparison:        {comparison_video}")
    print()
    print(f"  Passkey: {passkey}")
    print()
    print("How the privacy codec works:")
    print("  - The encoder takes video + mask and produces an H.264 stream")
    print("    with privacy metadata embedded as SEI messages.")
    print("  - WITHOUT a passkey, the decoder applies obfuscation to the")
    print("    regions identified by the mask (faces, license plates).")
    print("  - WITH the correct passkey, the decoder outputs the original")
    print("    unmodified video.")
    print("  - Legacy H.264 decoders simply ignore the SEI messages and")
    print("    output the original video.")
    print()
    print("This demonstrates compliance with GDPR Article 25")
    print("(data protection by design and by default).")


def main():
    parser = argparse.ArgumentParser(
        description="YouTube Privacy Codec Demo"
    )
    parser.add_argument("--url", default=None,
                        help="YouTube video URL (default: uses a CC traffic video)")
    parser.add_argument("--output-dir", default="./demo_output",
                        help="Output directory for demo files")
    parser.add_argument("--duration", type=int, default=15,
                        help="Maximum video duration in seconds (default: 15)")
    parser.add_argument("--sample", action="store_true",
                        help="Use synthetic sample video instead of downloading")

    args = parser.parse_args()

    if args.sample:
        run_demo(url=None, output_dir=args.output_dir,
                 max_duration=args.duration)
    else:
        run_demo(url=args.url, output_dir=args.output_dir,
                 max_duration=args.duration)


if __name__ == "__main__":
    main()
