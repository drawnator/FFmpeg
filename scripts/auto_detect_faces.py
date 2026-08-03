#!/usr/bin/env python3
"""
Automatic Face & License Plate Detection Mask Generator

Generates binary mask videos for privacy protection using YOLO-based
detection for faces and license plates.  The output mask is a grayscale
video where white regions (255) indicate detected sensitive areas.

Usage:
    # Detect faces only:
    python auto_detect_faces.py --input video.mp4 --output mask.raw \
        --detect faces

    # Detect faces and license plates:
    python auto_detect_faces.py --input video.mp4 --output mask.raw \
        --detect all

    # Full pipeline: detect + encode with privacy:
    python auto_detect_faces.py --input video.mp4 --output mask.raw \
        --detect all --encode --encoded-output privacy_video.mp4
"""

import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

# Try to import ultralytics for YOLO
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("Warning: ultralytics not installed. "
          "Install with: pip install ultralytics")


def get_ffmpeg_path():
    """Return path to the locally-built ffmpeg, falling back to system."""
    local = os.path.join(os.path.dirname(__file__), "..", "ffmpeg")
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local
    return "ffmpeg"


class FaceDetectorCV:
    """OpenCV DNN-based face detector (YuNet) as fallback."""

    def __init__(self):
        self.detector = None
        # Try to use YuNet via OpenCV
        yunet_path = os.path.join(
            os.path.dirname(__file__),
            "models", "face_detection_yunet_2023mar.onnx"
        )
        if os.path.isfile(yunet_path):
            self.detector = cv2.FaceDetectorYN.create(
                yunet_path, "", (320, 320),
                score_threshold=0.5,
                nms_threshold=0.3,
                top_k=5000,
            )
        else:
            # Fall back to Haar cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            self.detector = None

    def detect(self, frame):
        """
        Return a list of (x, y, w, h) bounding boxes for detected faces.
        """
        boxes = []
        h, w = frame.shape[:2]

        if self.detector is not None:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)
            if faces is not None:
                for face in faces:
                    x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    boxes.append((x, y, fw, fh))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for (x, y, fw, fh) in detections:
                boxes.append((x, y, fw, fh))

        return boxes


class YOLODetector:
    """YOLO-based detector for faces and license plates."""

    def __init__(self, detect_type="all"):
        self.detect_type = detect_type
        self.model = None
        self.face_detector = None

        if HAS_YOLO:
            # Use YOLOv8n for general object detection (includes cars for
            # licence plate region estimation)
            self.model = YOLO("yolov8n.pt")

            # For face detection, try yolov8n-face if available
            try:
                self.face_model = YOLO("yolov8n-face.pt")
            except Exception:
                self.face_model = None

        # Always have OpenCV face detector as fallback
        self.face_detector = FaceDetectorCV()

        # COCO class IDs for vehicles (used to estimate license plate regions)
        self.vehicle_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck

    def detect_faces(self, frame):
        """Detect faces using YOLO face model or OpenCV fallback."""
        boxes = []

        # Try YOLO face model first
        if self.face_model is not None:
            try:
                results = self.face_model(frame, verbose=False, conf=0.3)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        boxes.append((x1, y1, x2 - x1, y2 - y1))
                if boxes:
                    return boxes
            except Exception:
                pass

        # Fallback to OpenCV
        boxes = self.face_detector.detect(frame)
        return boxes

    def detect_license_plates(self, frame):
        """
        Detect license plate regions by finding vehicles and estimating
        plate location at the lower portion of vehicle bounding boxes.
        """
        boxes = []

        if self.model is None:
            return boxes

        results = self.model(frame, verbose=False, conf=0.4)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    vw = x2 - x1
                    vh = y2 - y1

                    # Estimate license plate region: lower center of vehicle
                    plate_h = max(int(vh * 0.15), 20)
                    plate_w = max(int(vw * 0.5), 40)
                    plate_x = x1 + (vw - plate_w) // 2
                    plate_y = y2 - int(vh * 0.25)

                    # Clamp to frame bounds
                    plate_x = max(0, plate_x)
                    plate_y = max(0, plate_y)

                    boxes.append((plate_x, plate_y, plate_w, plate_h))

        return boxes

    def detect(self, frame):
        """Return all detections as (x, y, w, h, label) tuples."""
        detections = []

        if self.detect_type in ("faces", "all"):
            for (x, y, w, h) in self.detect_faces(frame):
                # Add padding around face for better coverage
                pad_x = int(w * 0.2)
                pad_y = int(h * 0.2)
                detections.append((
                    max(0, x - pad_x),
                    max(0, y - pad_y),
                    w + 2 * pad_x,
                    h + 2 * pad_y,
                    "face"
                ))

        if self.detect_type in ("plates", "all"):
            for (x, y, w, h) in self.detect_license_plates(frame):
                # Add padding around plate
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)
                detections.append((
                    max(0, x - pad_x),
                    max(0, y - pad_y),
                    w + 2 * pad_x,
                    h + 2 * pad_y,
                    "plate"
                ))

        return detections


def generate_mask_video(input_video, output_mask, detect_type="all",
                        preview=False, skip_frames=1):
    """
    Process the input video and generate a raw grayscale mask file.

    Parameters
    ----------
    input_video : str
        Path to the input video.
    output_mask : str
        Path for the output raw mask file.
    detect_type : str
        What to detect: 'faces', 'plates', or 'all'.
    preview : bool
        If True, show a preview window during processing.
    skip_frames : int
        Run detection every N frames (interpolate between).
    """
    detector = YOLODetector(detect_type)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Input: {input_video}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.1f}, "
          f"Frames: {total_frames}")
    print(f"Detection type: {detect_type}")
    print(f"Skip frames: {skip_frames}")

    mask_fp = open(output_mask, "wb")

    frame_idx = 0
    last_detections = []
    total_faces = 0
    total_plates = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        if frame_idx % skip_frames == 0:
            detections = detector.detect(frame)
            last_detections = detections
        else:
            detections = last_detections

        # Create binary mask
        mask = np.zeros((height, width), dtype=np.uint8)

        for (x, y, w, h, label) in detections:
            # Clip to frame boundaries
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)

            mask[y1:y2, x1:x2] = 255

            if label == "face":
                total_faces += 1
            elif label == "plate":
                total_plates += 1

        # Write raw mask
        mask_fp.write(mask.tobytes())

        # Optional preview
        if preview:
            preview_frame = frame.copy()
            for (x, y, w, h, label) in detections:
                color = (0, 0, 255) if label == "face" else (255, 0, 0)
                cv2.rectangle(preview_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(preview_frame, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.imshow("Privacy Mask Preview", preview_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total_frames}: "
                  f"{len(detections)} detections")

    cap.release()
    mask_fp.close()

    if preview:
        cv2.destroyAllWindows()

    print(f"\nMask generation complete:")
    print(f"  Output: {output_mask}")
    print(f"  Total frames: {frame_idx}")
    print(f"  Total face detections: {total_faces}")
    print(f"  Total plate detections: {total_plates}")
    print(f"  Mask file size: {os.path.getsize(output_mask) / 1024 / 1024:.1f} MB")

    return width, height, fps


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate privacy masks using face/plate detection"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input video file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output raw mask file")
    parser.add_argument("--detect", choices=["faces", "plates", "all"],
                        default="all",
                        help="Detection type")
    parser.add_argument("--preview", action="store_true",
                        help="Show preview window during processing")
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Run detection every N frames (default: 1)")
    parser.add_argument("--encode", action="store_true",
                        help="Also encode the video with privacy metadata")
    parser.add_argument("--encoded-output", default=None,
                        help="Output path for privacy-encoded video")
    parser.add_argument("--obfuscation",
                        choices=["pixelate", "blur", "black", "invert"],
                        default="pixelate",
                        help="Obfuscation type for privacy mode")
    parser.add_argument("--no-encrypt", action="store_true",
                        help="Disable mask encryption")

    args = parser.parse_args()

    # Generate mask
    width, height, fps = generate_mask_video(
        args.input, args.output,
        detect_type=args.detect,
        preview=args.preview,
        skip_frames=args.skip_frames,
    )

    # Optionally encode with privacy metadata
    if args.encode:
        encoded_output = args.encoded_output or (args.input + ".privacy.mp4")
        print(f"\nEncoding with privacy metadata...")

        # Import and use the encoder
        sys.path.insert(0, os.path.dirname(__file__))
        from privacy_encode import encode_with_privacy, generate_key

        key_hex = None if args.no_encrypt else generate_key()
        if key_hex:
            print(f"Encryption key: {key_hex}")
            print("Save this key for authorized decoding!")

        # Create a temporary mask video from raw data
        tmpdir = tempfile.mkdtemp(prefix="privacy_mask_")
        mask_video = os.path.join(tmpdir, "mask.avi")

        # Convert raw mask to video
        ffmpeg = get_ffmpeg_path()
        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-pixel_format", "gray",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", args.output,
            "-c:v", "mjpeg",
            "-q:v", "2",
            mask_video,
        ]
        subprocess.run(cmd, capture_output=True, text=True)

        encode_with_privacy(
            args.input, mask_video, encoded_output,
            key_hex=key_hex,
            obfuscation=args.obfuscation,
            encryption_algo=0 if args.no_encrypt else 1,
        )

        # Create mask sidecar
        import shutil
        shutil.copy2(args.output, encoded_output + ".mask")
        print(f"Mask sidecar: {encoded_output}.mask")

        # Clean up
        try:
            os.remove(mask_video)
            os.rmdir(tmpdir)
        except OSError:
            pass


if __name__ == "__main__":
    main()
