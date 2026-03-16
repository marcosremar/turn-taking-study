"""
Deploy turn-taking benchmarks on Vast.ai GPU machines.

Uses BabelCast's existing Vast.ai infrastructure to provision machines,
upload benchmark code, run experiments, and collect results.

Usage:
    python deploy_vast.py --build          # Build and push Docker image
    python deploy_vast.py --deploy         # Deploy on Vast.ai
    python deploy_vast.py --run            # Run benchmarks on deployed machine
    python deploy_vast.py --collect        # Collect results
    python deploy_vast.py --cleanup        # Terminate instances
    python deploy_vast.py --all            # Do everything
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

STUDY_DIR = Path(__file__).parent
DOCKER_IMAGE = "marcosremar/babelcast-turn-taking-study:latest"

# Vast.ai API
VAST_API_BASE = "https://console.vast.ai/api/v0"


def get_vast_api_key() -> str:
    """Get Vast.ai API key from environment."""
    key = os.environ.get("VAST_API_KEY", "")
    if not key:
        env_path = STUDY_DIR.parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("VAST_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise RuntimeError("VAST_API_KEY not found in environment or .env")
    return key


def vast_api(method: str, endpoint: str, data: dict | None = None) -> dict:
    """Make a Vast.ai API call."""
    api_key = get_vast_api_key()
    url = f"{VAST_API_BASE}/{endpoint}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        log.error("Vast.ai API error %d: %s", e.code, error_body)
        raise


def build_docker_image() -> None:
    """Build and push Docker image for benchmarks."""
    log.info("Building Docker image: %s", DOCKER_IMAGE)

    subprocess.run(
        ["docker", "build", "-t", DOCKER_IMAGE, "-f", str(STUDY_DIR / "Dockerfile"), str(STUDY_DIR)],
        check=True,
    )
    log.info("Pushing Docker image...")
    subprocess.run(["docker", "push", DOCKER_IMAGE], check=True)
    log.info("Image pushed: %s", DOCKER_IMAGE)


def find_gpu_offer(gpu_type: str = "RTX A6000", min_ram_gb: int = 16) -> dict | None:
    """Find a suitable Vast.ai GPU offer."""
    log.info("Searching for %s with >= %dGB RAM...", gpu_type, min_ram_gb)

    # Search for offers
    result = vast_api("GET", f"bundles?q={{\"gpu_name\":\"{gpu_type}\",\"gpu_ram\":{{\">=\":{min_ram_gb}}},\"rentable\":{{\"eq\":true}},\"order\":[[\"dph_total\",\"asc\"]],\"type\":\"on-demand\"}}")

    offers = result.get("offers", [])
    if not offers:
        log.warning("No %s offers found, trying RTX 4090...", gpu_type)
        result = vast_api("GET", "bundles?q={\"gpu_name\":\"RTX 4090\",\"rentable\":{\"eq\":true},\"order\":[[\"dph_total\",\"asc\"]],\"type\":\"on-demand\"}")
        offers = result.get("offers", [])

    if offers:
        offer = offers[0]
        log.info("Found: %s @ $%.3f/hr (ID: %s)", offer.get("gpu_name"), offer.get("dph_total", 0), offer.get("id"))
        return offer

    return None


def deploy_instance(offer_id: int) -> dict:
    """Deploy a Vast.ai instance with the benchmark Docker image."""
    log.info("Deploying instance on offer %s...", offer_id)

    result = vast_api("PUT", f"asks/{offer_id}/", data={
        "client_id": "me",
        "image": DOCKER_IMAGE,
        "disk": 30,  # GB
        "onstart": "cd /workspace/turn-taking-study && python run_benchmarks.py --all 2>&1 | tee /workspace/benchmark.log",
        "runtype": "args",
        "env": {
            "HF_HUB_CACHE": "/workspace/hf_cache",
            "PYTHONUNBUFFERED": "1",
        },
    })

    instance_id = result.get("new_contract")
    log.info("Instance deployed: %s", instance_id)
    return result


def wait_for_instance(instance_id: int, timeout_min: int = 15) -> dict:
    """Wait for instance to be ready."""
    log.info("Waiting for instance %s to be ready...", instance_id)
    deadline = time.time() + timeout_min * 60
    poll_interval = 10

    while time.time() < deadline:
        result = vast_api("GET", f"instances/{instance_id}/")
        status = result.get("actual_status", "unknown")
        log.info("Instance %s status: %s", instance_id, status)

        if status == "running":
            return result
        if status in ("error", "exited"):
            raise RuntimeError(f"Instance failed with status: {status}")

        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.3, 30)

    raise TimeoutError(f"Instance {instance_id} did not become ready in {timeout_min}min")


def collect_results(instance_id: int) -> dict:
    """Download benchmark results from instance."""
    log.info("Collecting results from instance %s...", instance_id)

    instance = vast_api("GET", f"instances/{instance_id}/")
    ssh_host = instance.get("ssh_host", "")
    ssh_port = instance.get("ssh_port", 22)

    if not ssh_host:
        log.error("No SSH access available for instance %s", instance_id)
        return {}

    results_dir = STUDY_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Download results via SCP
    subprocess.run([
        "scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no",
        f"root@{ssh_host}:/workspace/turn-taking-study/results/*.json",
        str(results_dir),
    ], check=False)

    # Download log
    subprocess.run([
        "scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no",
        f"root@{ssh_host}:/workspace/benchmark.log",
        str(STUDY_DIR / "benchmark.log"),
    ], check=False)

    log.info("Results collected in %s", results_dir)
    return {"results_dir": str(results_dir)}


def cleanup_instance(instance_id: int) -> None:
    """Terminate a Vast.ai instance."""
    log.info("Terminating instance %s...", instance_id)
    vast_api("DELETE", f"instances/{instance_id}/")
    log.info("Instance %s terminated", instance_id)


def deploy_via_gateway() -> dict | None:
    """
    Alternative: Deploy via BabelCast gateway (uses existing Vast.ai integration).
    Requires gateway running on localhost:4000.
    """
    import urllib.request
    import json

    body = json.dumps({
        "dockerImage": DOCKER_IMAGE,
        "gpuTypes": ["NVIDIA RTX A6000"],
    }).encode()

    req = urllib.request.Request(
        "http://localhost:4000/v1/gpu/deploy",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            log.info("Deployed via gateway: %s", result)
            return result
    except Exception as e:
        log.warning("Gateway deploy failed: %s — falling back to direct Vast.ai API", e)
        return None


def run_all(gpu_type: str = "RTX A6000") -> None:
    """Run the full benchmark pipeline."""
    state_file = STUDY_DIR / ".deploy_state.json"

    # Step 1: Build
    log.info("=== Step 1: Build Docker Image ===")
    build_docker_image()

    # Step 2: Deploy
    log.info("=== Step 2: Deploy on Vast.ai ===")
    offer = find_gpu_offer(gpu_type)
    if not offer:
        raise RuntimeError("No GPU offers available")

    result = deploy_instance(offer["id"])
    instance_id = result.get("new_contract")

    # Save state
    with open(state_file, "w") as f:
        json.dump({"instance_id": instance_id, "offer": offer}, f, indent=2)

    # Step 3: Wait
    log.info("=== Step 3: Wait for Instance ===")
    instance = wait_for_instance(instance_id)

    # Step 4: Wait for benchmarks to complete
    log.info("=== Step 4: Waiting for benchmarks (check logs) ===")
    log.info("Monitor with: vast logs %s", instance_id)
    log.info("Benchmarks typically take 20-40 minutes depending on GPU")

    # Poll for completion
    for _ in range(60):  # Up to 60 minutes
        time.sleep(60)
        try:
            inst = vast_api("GET", f"instances/{instance_id}/")
            if inst.get("actual_status") == "exited":
                log.info("Benchmarks completed!")
                break
        except Exception:
            continue

    # Step 5: Collect
    log.info("=== Step 5: Collect Results ===")
    collect_results(instance_id)

    # Step 6: Cleanup
    log.info("=== Step 6: Cleanup ===")
    cleanup_instance(instance_id)

    if state_file.exists():
        state_file.unlink()

    log.info("=== Done! Results in %s ===", STUDY_DIR / "results")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Deploy turn-taking benchmarks on Vast.ai")
    parser.add_argument("--build", action="store_true", help="Build and push Docker image")
    parser.add_argument("--deploy", action="store_true", help="Deploy instance")
    parser.add_argument("--run", action="store_true", help="Run benchmarks on deployed instance")
    parser.add_argument("--collect", action="store_true", help="Collect results")
    parser.add_argument("--cleanup", action="store_true", help="Terminate instance")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--gpu", default="RTX A6000", help="GPU type (default: RTX A6000)")
    parser.add_argument("--instance-id", type=int, help="Instance ID for collect/cleanup")
    args = parser.parse_args()

    if args.all:
        run_all(args.gpu)
    elif args.build:
        build_docker_image()
    elif args.deploy:
        offer = find_gpu_offer(args.gpu)
        if offer:
            deploy_instance(offer["id"])
    elif args.collect and args.instance_id:
        collect_results(args.instance_id)
    elif args.cleanup and args.instance_id:
        cleanup_instance(args.instance_id)
    else:
        parser.print_help()
