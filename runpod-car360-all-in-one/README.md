# RunPod Car360 — all in one

This repo is designed for a **RunPod GPU Pod**, not RunPod Serverless.

It uses:
- `morphicfilms/frames-to-video`
- `Wan-AI/Wan2.2-I2V-A14B`
- `morphic/Wan2.2-frames-to-video`

The upstream Morphic workflow supports start image, end image and multiple middle images. This app maps the car views as:

- 0.00 — Front
- 0.25 — Left
- 0.50 — Rear
- 0.75 — Right
- 1.00 — Front again

It generates 113 frames at 832×480, then FFmpeg retimes the result to exactly 15 seconds and interpolates to 30fps.

## Recommended RunPod GPU

Use:
- A100 80GB, or
- H100 80GB

The upstream Wan2.2 I2V A14B model is very large. An 80GB GPU is the straightforward setup.

## RunPod setup

### 1. Put this repository on GitHub

Upload all files in this folder to a new GitHub repository.

### 2. Build a Docker image

Use GitHub Actions / Docker Hub / RunPod image builder, or build:

```bash
docker build -t YOUR_DOCKERHUB_USERNAME/car360:latest .
docker push YOUR_DOCKERHUB_USERNAME/car360:latest
```

### 3. Create a RunPod Pod

Create a GPU Pod with:

- GPU: A100 80GB or H100 80GB
- Container image: your `car360:latest`
- Container disk: at least 30GB
- Persistent volume: **at least 100GB**
- Volume mount path: `/workspace`
- Expose HTTP port: `8000`

### 4. Wait for the first model download

Open the Pod logs.

On first launch it downloads:
- Wan2.2 I2V A14B base weights
- Morphic frames-to-video LoRA

They are stored under:

```text
/workspace/models
```

With persistent storage, later restarts reuse them.

### 5. Open the web UI

In RunPod click the HTTP service for port `8000`.

Upload:
- Front
- Left
- Rear
- Right

and press **Generate 360 video**.

## Output

The final output is:

```text
/workspace/car360-output/<job>/car360-15sec-480p.mp4
```

The UI also gives:
- final 15-second MP4
- raw Wan video
- generation log

## Why this is different from the earlier API templates

This does **not** send a 2×2 reference sheet into a one-image API.

It directly uses the multi-frame interpolation support added by Morphic:

```text
front -> left -> rear -> right -> front
```

with timestamps `0.25 0.50 0.75`.

## Model sources

Morphic repository:
https://github.com/morphicfilms/frames-to-video

Wan2.2 model:
https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B

Morphic LoRA:
https://huggingface.co/morphic/Wan2.2-frames-to-video
