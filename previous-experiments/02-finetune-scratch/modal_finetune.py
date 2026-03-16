"""Deploy fine-tuning on Modal.

Modal supports custom Docker images, GPU selection, and long-running jobs.
This bypasses the gateway's deploy pipeline since it only supports the
translation pipeline on Modal, not arbitrary workloads.
"""

import modal
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

app = modal.App("babelcast-finetune-smart-turn-v3-focal")

# Docker image with PyTorch + deps + training script baked in
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchaudio",
        "transformers",
        "datasets>=2.18,<3.0",
        "soundfile",
        "librosa",
        "numpy",
    )
    .apt_install("ffmpeg", "libsndfile1")
    .add_local_file(
        str(SCRIPT_DIR / "finetune_smart_turn_v3.py"),
        remote_path="/root/finetune_smart_turn_v3.py",
    )
)

# Persistent volume for checkpoints + HF cache
vol = modal.Volume.from_name("finetune-smart-turn-v3-focal", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # cheapest Modal GPU, 24GB VRAM — sufficient for Whisper Tiny fine-tuning
    timeout=4 * 3600,  # 4 hours max
    volumes={"/workspace": vol},
)
def run_finetune():
    """Run the fine-tuning script on a GPU."""
    import subprocess
    import sys
    import os

    os.environ["HF_HOME"] = "/workspace/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"

    # Copy baked-in script to workspace
    script_dest = Path("/workspace/finetune_smart_turn_v3.py")
    script_dest.write_text(Path("/root/finetune_smart_turn_v3.py").read_text())

    # Check for existing checkpoint
    ckpt = Path("/workspace/checkpoints/smart_turn_pt_v3/resume_checkpoint.pt")
    if ckpt.exists():
        print("[modal] Found resume checkpoint — continuing training")
    else:
        print("[modal] Starting fresh training")

    # Run training
    print("[modal] Starting fine-tuning...")
    result = subprocess.run(
        [sys.executable, str(script_dest)],
        cwd="/workspace",
        env={**os.environ},
    )

    # Commit volume changes (checkpoints, results)
    vol.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    # List results
    results_dir = Path("/workspace/checkpoints/smart_turn_pt_v3")
    if results_dir.exists():
        print("\n[modal] Training results:")
        for f in sorted(results_dir.iterdir()):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name}: {size_mb:.1f} MB")

    return "Training complete!"


@app.function(
    image=image,
    volumes={"/workspace": vol},
)
def check_status():
    """Check training status (checkpoint existence, results)."""
    from pathlib import Path

    results_dir = Path("/workspace/checkpoints/smart_turn_pt_v3")
    status = {
        "has_checkpoint": (results_dir / "resume_checkpoint.pt").exists(),
        "done": (results_dir / "training_results.json").exists(),
    }

    if status["done"]:
        import json
        results = json.loads((results_dir / "training_results.json").read_text())
        status["results"] = results

    if results_dir.exists():
        status["files"] = [f.name for f in sorted(results_dir.iterdir())]

    return status


@app.function(
    image=image,
    volumes={"/workspace": vol},
)
def download_results() -> dict:
    """Download training results from the volume."""
    results_dir = Path("/workspace/checkpoints/smart_turn_pt_v3")

    files = {}
    for f in results_dir.iterdir():
        if f.suffix in (".json", ".txt"):
            files[f.name] = f.read_text()
        elif f.suffix in (".onnx", ".pt"):
            files[f.name] = f"[binary, {f.stat().st_size / 1024 / 1024:.1f} MB]"

    return files


@app.local_entrypoint()
def main():
    print("Starting fine-tuning on Modal...")
    print(f"Script: {SCRIPT_DIR / 'finetune_smart_turn_v3.py'}")
    print(f"GPU: A10G (24GB VRAM)")
    print(f"Timeout: 4 hours")
    print()

    result = run_finetune.remote()
    print(f"\nResult: {result}")
