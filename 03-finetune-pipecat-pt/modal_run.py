"""Run the full fine-tuning pipeline on Modal (A10G GPU).

Usage:
    modal run modal_run.py

This deploys the pipeline on a Modal A10G GPU (~$0.50/run):
1. Downloads Pipecat Portuguese data
2. Generates labels with Claude API (if ANTHROPIC_API_KEY set)
3. Generates TTS audio with Kokoro
4. Fine-tunes SmartTurnV3Model
5. Evaluates against Pipecat baseline
6. Saves results to Modal volume

Estimated time: 30-60 min total
Estimated cost: ~$0.50-1.00
"""

from __future__ import annotations

import modal

app = modal.App("babelcast-finetune-pipecat-pt")

# Persistent volume for data + results
volume = modal.Volume.from_name("finetune-pipecat-pt", create_if_missing=True)

# GPU image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core ML
        "torch>=2.1",
        "torchaudio>=2.1",
        "transformers>=4.36",
        "datasets>=2.16",
        # Audio processing
        "soundfile>=0.12",
        "librosa>=0.10",
        "numpy<2",
        # TTS
        "kokoro>=0.3",
        # ONNX for evaluation
        "onnxruntime>=1.16",
        # Claude API for labeling
        "anthropic>=0.40",
        # Utilities
        "huggingface-hub>=0.20",
    )
    .apt_install("ffmpeg", "libsndfile1")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,  # 1 hour max
    volumes={"/workspace": volume},
    secrets=[
        modal.Secret.from_name("anthropic-api-key", required=False),
        modal.Secret.from_name("huggingface-token", required=False),
    ],
)
def run_pipeline(
    skip_download: bool = False,
    skip_labels: bool = False,
    skip_audio: bool = False,
    skip_train: bool = False,
    max_pipecat_samples: int = 5000,
    max_tts_samples: int = 10000,
    max_l2_samples: int = 2000,
    epochs: int = 6,
    batch_size: int = 128,
    lr: float = 5e-5,
    loss_fn: str = "focal",
    fp_penalty: float = 2.0,
):
    """Run the full pipeline on Modal GPU."""
    import logging
    import os
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    log = logging.getLogger("modal_run")

    # Add script directory to path
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))

    # Symlink data dir to /workspace for persistence
    data_dir = script_dir / "data"
    workspace_data = Path("/workspace/data")
    workspace_data.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        data_dir.symlink_to(workspace_data)

    # Step 1: Download Pipecat data
    if not skip_download:
        log.info("=" * 60)
        log.info("STEP 1: Download Pipecat Portuguese data")
        log.info("=" * 60)
        from import_module_01 import download_pipecat_dataset, download_pipecat_test_data, download_onnx_model
        try:
            # Use importlib since filenames start with numbers
            import importlib.util
            spec = importlib.util.spec_from_file_location("dl", script_dir / "01_download_pipecat.py")
            dl = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dl)

            dl.download_pipecat_dataset(max_pt_samples=max_pipecat_samples)
            dl.download_pipecat_test_data(max_pt_samples=2000)
            dl.download_onnx_model()
        except Exception as e:
            log.error("Download failed: %s", e)
            raise

        volume.commit()
        log.info("Step 1 complete — data saved to volume")

    # Step 2: Generate labels (requires ANTHROPIC_API_KEY)
    if not skip_labels:
        log.info("=" * 60)
        log.info("STEP 2: Generate labels with Claude API")
        log.info("=" * 60)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("labels", script_dir / "02_generate_labels.py")
            labels = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(labels)

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                log.info("ANTHROPIC_API_KEY found — using Claude for labeling")
            else:
                log.info("No ANTHROPIC_API_KEY — using rule-based fallback")

            labels.run_full_pipeline(max_transcripts=5000, max_fr_sentences=500)
        except Exception as e:
            log.warning("Label generation failed: %s — continuing without custom labels", e)

        volume.commit()

    # Step 3: Generate TTS audio
    if not skip_audio:
        log.info("=" * 60)
        log.info("STEP 3: Generate TTS audio")
        log.info("=" * 60)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("audio", script_dir / "03_generate_audio.py")
            audio = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(audio)

            audio.run_audio_generation(
                max_native_sentences=3000,
                max_french_sentences=1000,
                augment_copies=2,
            )
        except Exception as e:
            log.warning("TTS generation failed: %s — continuing with Pipecat data only", e)

        volume.commit()

    # Step 4: Fine-tune
    if not skip_train:
        log.info("=" * 60)
        log.info("STEP 4: Fine-tune SmartTurnV3Model")
        log.info("=" * 60)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("ft", script_dir / "04_finetune.py")
            ft = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ft)

            ft.train(
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                warmup_ratio=0.2,
                max_pipecat_samples=max_pipecat_samples,
                max_tts_samples=max_tts_samples,
                max_l2_samples=max_l2_samples,
                loss_fn=loss_fn,
                fp_penalty=fp_penalty,
            )
        except Exception as e:
            log.error("Training failed: %s", e)
            raise

        volume.commit()

    # Step 5: Evaluate
    log.info("=" * 60)
    log.info("STEP 5: Evaluate")
    log.info("=" * 60)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("ev", script_dir / "05_evaluate.py")
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)

        results_dir = Path("/workspace/results")
        model_path = results_dir / "best_model.pt"
        if model_path.exists():
            results = ev.run_evaluation(model_path)
            log.info("Evaluation complete!")
        else:
            log.warning("No model found at %s — skipping evaluation", model_path)
    except Exception as e:
        log.error("Evaluation failed: %s", e)

    volume.commit()
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE — results saved to Modal volume 'finetune-pipecat-pt'")
    log.info("=" * 60)


@app.function(
    image=image,
    volumes={"/workspace": volume},
)
def download_results(local_dir: str = "results") -> list[str]:
    """Download results from Modal volume."""
    from pathlib import Path
    import shutil

    results_dir = Path("/workspace/results")
    if not results_dir.exists():
        print("No results found on volume")
        return []

    files = []
    for f in results_dir.rglob("*"):
        if f.is_file():
            files.append(str(f.relative_to(results_dir)))
            print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")

    return files


@app.local_entrypoint()
def main(
    skip_download: bool = False,
    skip_labels: bool = False,
    skip_audio: bool = False,
    skip_train: bool = False,
    download_only: bool = False,
    epochs: int = 10,
    batch_size: int = 32,
):
    """Entry point for `modal run modal_run.py`."""
    if download_only:
        files = download_results.remote()
        print(f"\nFound {len(files)} result files on volume")
        return

    run_pipeline.remote(
        skip_download=skip_download,
        skip_labels=skip_labels,
        skip_audio=skip_audio,
        skip_train=skip_train,
        epochs=epochs,
        batch_size=batch_size,
    )
