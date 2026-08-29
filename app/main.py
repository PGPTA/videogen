import asyncio
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter, ImageDraw

APP_DIR = Path("/opt/car360")
ENGINE_DIR = Path("/opt/frames-to-video")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/workspace/models"))
OUT_ROOT = Path("/workspace/car360-output")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

BASE_MODEL = MODEL_DIR / "Wan2.2-I2V-A14B-Interpolation"
LORA = MODEL_DIR / "Wan2.2-frames-to-video" / "lora_interpolation_high_noise_final.safetensors"

app = FastAPI(title="Car360 RunPod")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

jobs: Dict[str, dict] = {}

PROMPT = """A photorealistic studio product video of the exact same car.
The supplied images are fixed keyframes of the same vehicle in clockwise order:
front view, left side, rear view, right side, then back to front.

Interpolate naturally between those exact views as if a camera makes one smooth,
constant-speed 360 degree orbit around a completely stationary vehicle.

Preserve vehicle identity exactly: body shape, paint, bumpers, grille, headlights,
taillights, wheels, tyres, mirrors, badges, number plate, glass tint, trim,
ride height and proportions. No redesigning and no morphing.

Pure black studio background. Subtle soft elliptical drop shadow directly below
the vehicle. No scenery, no people, no text, no extra objects, no glossy floor
reflection. Car remains centered. Constant camera height and distance. No zoom.
No cuts. No wheel rotation. First and final front views must align for a loop."""

def update(job_id, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)

def save_upload(upload: UploadFile, dest: Path):
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)

def prep_image(src: Path, dst: Path):
    im = Image.open(src).convert("RGBA")
    bbox = im.getbbox()
    if bbox:
        im = im.crop(bbox)

    canvas_w, canvas_h = 832, 480
    max_w, max_h = 760, 405
    im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))

    shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(shadow)
    d.ellipse(
        (canvas_w//2 - 220, canvas_h - 52, canvas_w//2 + 220, canvas_h - 18),
        fill=(145, 145, 145, 65),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(15))
    canvas = Image.alpha_composite(canvas, shadow)

    x = (canvas_w - im.width) // 2
    y = max(5, (canvas_h - im.height) // 2 - 12)
    canvas.alpha_composite(im, (x, y))
    canvas.convert("RGB").save(dst, quality=96)

def run_command(cmd, cwd: Path, log_path: Path):
    with log_path.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return p.wait()

def process_video(raw: Path, final: Path):
    probe = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(raw)
    ], text=True).strip()
    duration = float(probe)
    factor = 15.0 / duration

    subprocess.check_call([
        "ffmpeg", "-y", "-i", str(raw),
        "-vf",
        f"setpts={factor:.10f}*PTS,"
        "minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
        "scale=832:480:flags=lanczos",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-t", "15",
        str(final)
    ])

def generate_job(job_id: str, job_dir: Path, seed: int, steps: int):
    started = time.time()
    try:
        update(job_id, status="running", phase="prepare", progress=5,
               message="Preparing the four views on black…")

        prepared = {}
        for name in ("front", "left", "rear", "right"):
            src = job_dir / f"{name}.upload"
            dst = job_dir / f"{name}.png"
            prep_image(src, dst)
            prepared[name] = dst

        raw = job_dir / "orbit-raw.mp4"
        final = job_dir / "car360-15sec-480p.mp4"
        log = job_dir / "generation.log"

        update(job_id, phase="model", progress=12,
               message="Loading Wan2.2 and the multi-frame interpolation LoRA…",
               elapsed=int(time.time()-started))

        cmd = [
            "python3", "generate.py",
            "--task", "i2v-A14B",
            "--size", "832*480",
            "--frame_num", "113",
            "--ckpt_dir", str(BASE_MODEL),
            "--high_noise_lora_weights_path", str(LORA),
            "--image", str(prepared["front"]),
            "--middle_images",
            str(prepared["left"]),
            str(prepared["rear"]),
            str(prepared["right"]),
            "--middle_images_timestamps", "0.25", "0.50", "0.75",
            "--img_end", str(prepared["front"]),
            "--prompt", PROMPT,
            "--sample_steps", str(steps),
            "--base_seed", str(seed),
            "--offload_model", "True",
            "--convert_model_dtype",
            "--t5_cpu",
            "--save_file", str(raw),
        ]

        # Generation progress is estimated because the upstream script does not expose a callback API.
        p = subprocess.Popen(
            cmd,
            cwd=str(ENGINE_DIR),
            stdout=log.open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            text=True,
        )

        while p.poll() is None:
            elapsed = int(time.time() - started)
            progress = min(84, 18 + elapsed // 12)
            update(job_id, phase="generate", progress=progress,
                   message="Generating the 360° orbit with Wan2.2…",
                   elapsed=elapsed)
            time.sleep(5)

        if p.returncode != 0:
            tail = log.read_text(errors="ignore")[-6000:]
            raise RuntimeError("Wan generation failed.\n\n" + tail)

        if not raw.exists() or raw.stat().st_size < 10000:
            tail = log.read_text(errors="ignore")[-5000:]
            raise RuntimeError("Wan finished but no valid MP4 was produced.\n\n" + tail)

        update(job_id, phase="finish", progress=90,
               message="Making the final 15-second 30fps MP4…",
               elapsed=int(time.time()-started))

        process_video(raw, final)

        update(
            job_id,
            status="done",
            phase="done",
            progress=100,
            message="Finished.",
            elapsed=int(time.time()-started),
            video=f"/api/jobs/{job_id}/video",
            raw=f"/api/jobs/{job_id}/raw",
            log=f"/api/jobs/{job_id}/log",
            seed=seed,
            steps=steps,
        )
    except Exception as e:
        update(
            job_id,
            status="error",
            phase="error",
            progress=0,
            message=str(e),
            elapsed=int(time.time()-started),
            log=f"/api/jobs/{job_id}/log",
        )

@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "base_model": BASE_MODEL.exists(),
        "lora": LORA.exists(),
        "engine": ENGINE_DIR.exists(),
    }

@app.post("/api/generate")
async def create_job(
    front: UploadFile = File(...),
    left: UploadFile = File(...),
    rear: UploadFile = File(...),
    right: UploadFile = File(...),
    seed: int = Form(-1),
    steps: int = Form(20),
):
    if not BASE_MODEL.exists():
        raise HTTPException(503, "Wan2.2 model is still downloading. Check the RunPod logs.")
    if not LORA.exists():
        raise HTTPException(503, "Frames-to-video LoRA is still downloading. Check the RunPod logs.")

    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "big")
    steps = max(8, min(int(steps), 40))

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUT_ROOT / job_id
    job_dir.mkdir(parents=True)

    for name, upload in {
        "front": front,
        "left": left,
        "rear": rear,
        "right": right,
    }.items():
        save_upload(upload, job_dir / f"{name}.upload")

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "phase": "queued",
        "progress": 1,
        "message": "Starting…",
        "elapsed": 0,
    }

    asyncio.get_running_loop().run_in_executor(
        None, generate_job, job_id, job_dir, seed, steps
    )

    return {"job_id": job_id}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]

@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str):
    p = OUT_ROOT / job_id / "car360-15sec-480p.mp4"
    if not p.exists():
        raise HTTPException(404, "Final video not found")
    return FileResponse(p, media_type="video/mp4", filename="car360-15sec-480p.mp4")

@app.get("/api/jobs/{job_id}/raw")
def get_raw(job_id: str):
    p = OUT_ROOT / job_id / "orbit-raw.mp4"
    if not p.exists():
        raise HTTPException(404, "Raw video not found")
    return FileResponse(p, media_type="video/mp4", filename="car360-raw.mp4")

@app.get("/api/jobs/{job_id}/log")
def get_log(job_id: str):
    p = OUT_ROOT / job_id / "generation.log"
    if not p.exists():
        raise HTTPException(404, "Log not found")
    return FileResponse(p, media_type="text/plain", filename="generation.log")
