# Privacy-Preserving H.264 Codec Extension

## Overview

This extension adds selective privacy protection to FFmpeg's H.264 codec
implementation, based on the Privacy Protection Profile (PPP) described in
the paper "Towards a Privacy-Preserving Video Codec: A Selective Encryption
Framework for H.264."

The system processes two correlated streams—a primary video stream and a binary
mask stream—to enable conditional reconstruction:

- **Without a passkey** → privacy-sensitive regions (faces, license plates) are
  obfuscated (pixelated, blurred, blacked out, or color-inverted).
- **With the correct passkey** → the original video is reconstructed in full.
- **Legacy decoders** → ignore the SEI privacy metadata and output the original
  video unchanged (backward compatible).

## Architecture

```
                  ┌──────────────┐
                  │  Input Video │
                  └──────┬───────┘
                         │
              ┌──────────┴──────────┐
              │                     │
     ┌────────▼────────┐   ┌───────▼────────┐
     │  H.264 Encoder  │   │ Face/Plate     │
     │  (libx264)      │   │ Detection      │
     └────────┬────────┘   │ (YOLO/OpenCV)  │
              │            └───────┬────────┘
              │                    │
              │            ┌───────▼────────┐
              │            │ Mask Generator │
              │            │ (binary mask)  │
              │            └───────┬────────┘
              │                    │
     ┌────────▼────────────────────▼────────┐
     │       h264_privacy BSF               │
     │  (RLE compress → encrypt → SEI NAL)  │
     └────────────────┬─────────────────────┘
                      │
             ┌────────▼────────┐
             │  Output H.264   │
             │  + Privacy SEI  │
             └────────┬────────┘
                      │
         ┌────────────┴───────────────┐
         │                            │
┌────────▼────────┐         ┌────────▼────────┐
│ Decode w/o key  │         │ Decode w/ key   │
│ (privacy mode)  │         │ (authorized)    │
│ → Obfuscated    │         │ → Original      │
└─────────────────┘         └─────────────────┘
```

## Components

### 1. SEI Message Type (`SEI_TYPE_PRIVACY_METADATA` = 210)

Defined in `libavcodec/sei.h`. This new SEI payload type carries encrypted
privacy mask data within the H.264 bitstream.

**Payload format:**

| Field                     | Size      | Description                                |
|---------------------------|-----------|--------------------------------------------|
| privacy_metadata_id       | ue(v)     | Instance identifier                        |
| privacy_metadata_cancel   | u(1)      | Cancel previous metadata persistence       |
| mask_encryption_algorithm | u(8)      | 0=none, 1=AES-128-CBC, 2=AES-256-CBC      |
| key_id                    | u(128)    | 16-byte decryption key identifier          |
| obfuscation_type          | u(8)      | 0=pixelate, 1=blur, 2=black, 3=invert     |
| encrypted_mask_data       | b(8)      | RLE-compressed mask (optionally encrypted) |

### 2. H264 Privacy BSF (`h264_privacy`)

Bitstream filter in `libavcodec/bsf/h264_privacy.c` that injects privacy
metadata SEI NAL units into an H.264 bitstream.

**Options:**

| Option        | Type   | Description                                    |
|---------------|--------|------------------------------------------------|
| mask_file     | string | Path to raw binary mask (grayscale frames)     |
| obfuscation   | int    | 0=pixelate, 1=blur, 2=black, 3=color_invert   |
| encryption    | int    | 0=none, 1=AES-128, 2=AES-256                  |
| key           | string | Hex-encoded encryption key                     |
| metadata_id   | int    | Privacy metadata instance identifier           |

**Example:**

```bash
ffmpeg -i input.mp4 -c:v libx264 -bsf:v \
    "h264_privacy=mask_file=mask.raw:obfuscation=0" \
    output.h264
```

### 3. Privacy Mask Video Filter (`privacy_mask`)

Video filter in `libavfilter/vf_privacy_mask.c` that applies obfuscation
to regions identified by a binary mask file.

**Options:**

| Option         | Type   | Description                               |
|----------------|--------|-------------------------------------------|
| mask_file      | string | Path to raw grayscale mask video           |
| passkey        | string | Passkey for authorized mode (no obfusc.)  |
| obfuscation    | int    | 0=pixelate, 1=blur, 2=black, 3=invert    |
| block_size     | int    | Pixelation block size (default: 16)       |
| blur_strength  | int    | Blur kernel radius (default: 8)           |

**Example:**

```bash
# Privacy mode (obfuscated output):
ffmpeg -i input.mp4 -vf "privacy_mask=mask_file=mask.raw:obfuscation=0" \
    censored.mp4

# Authorized mode (original output):
ffmpeg -i input.mp4 -vf "privacy_mask=passkey=abc123" original.mp4
```

## Python Scripts

### `scripts/privacy_encode.py`

Encode a video with privacy mask metadata.

```bash
python scripts/privacy_encode.py \
    --input video.mp4 \
    --mask mask_video.mp4 \
    --output privacy_encoded.mp4 \
    --obfuscation pixelate
```

### `scripts/privacy_decode.py`

Decode a privacy-encoded video.

```bash
# Censored output (no passkey):
python scripts/privacy_decode.py \
    --input privacy_encoded.mp4 \
    --output censored.mp4

# Original output (with passkey):
python scripts/privacy_decode.py \
    --input privacy_encoded.mp4 \
    --output original.mp4 \
    --passkey <hex_key>
```

### `scripts/auto_detect_faces.py`

Automatically detect faces and license plates to generate mask files.

```bash
# Detect faces and license plates:
python scripts/auto_detect_faces.py \
    --input video.mp4 \
    --output mask.raw \
    --detect all

# Detect faces only with full encode pipeline:
python scripts/auto_detect_faces.py \
    --input video.mp4 \
    --output mask.raw \
    --detect faces \
    --encode \
    --encoded-output privacy_video.mp4
```

### `scripts/demo_youtube.py`

Full demonstration using a YouTube video.

```bash
# Run with synthetic sample video:
python scripts/demo_youtube.py --sample --output-dir ./demo_output

# Run with a specific YouTube video:
python scripts/demo_youtube.py --url "https://youtube.com/watch?v=..." \
    --output-dir ./demo_output --duration 15
```

## Quick Start

### Prerequisites

```bash
pip install -r scripts/requirements.txt
```

### Build FFmpeg with Privacy Extensions

```bash
./configure --enable-gpl --enable-libx264
make -j$(nproc)
```

### Run the Demo

```bash
# Full demo with synthetic video (no internet required):
python scripts/demo_youtube.py --sample

# Full demo with YouTube video:
python scripts/demo_youtube.py --duration 15
```

### Manual Pipeline

```bash
# 1. Generate masks automatically
python scripts/auto_detect_faces.py \
    -i surveillance_footage.mp4 \
    -o masks.raw \
    --detect all

# 2. Encode with privacy metadata
python scripts/privacy_encode.py \
    -i surveillance_footage.mp4 \
    -m mask_video.mp4 \
    -o encoded.mp4

# 3. Decode (censored)
python scripts/privacy_decode.py \
    -i encoded.mp4 \
    -o censored.mp4

# 4. Decode (authorized)
python scripts/privacy_decode.py \
    -i encoded.mp4 \
    -o original.mp4 \
    --passkey <key_from_step_2>
```

## Detection Models

The auto-detection pipeline supports:

| Model           | Purpose               | Fallback             |
|-----------------|-----------------------|----------------------|
| YOLOv8n         | General object detect | N/A                  |
| YOLOv8n-face    | Face detection        | OpenCV Haar cascades |
| YuNet (OpenCV)  | Face detection        | Haar cascades        |

License plate detection works by detecting vehicles (cars, buses, trucks,
motorcycles) and estimating the plate region in the lower portion of each
vehicle bounding box.

## Privacy Compliance

This framework supports:

- **GDPR Article 5(1)(b)**: Selective access based on authorization
- **GDPR Article 25**: Data protection by design and by default
- **EU AI Act Article 5**: Controlled biometric identification access

## Limitations

- Detection accuracy depends on the model and input quality
- Mask data adds 2-5% bitrate overhead (RLE compressed)
- Temporal consistency (flickering) may occur frame-to-frame
- Key management is out-of-band (not included in the stream)
