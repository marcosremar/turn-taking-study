"""Deploy fine-tuning pod via the AI Gateway.

Strategy: Use RunPod's PyTorch template image (has SSH + PyTorch pre-installed).
Write the training script directly from Python (no base64 env var size limits).
"""
import json
import urllib.request
from pathlib import Path

GATEWAY_URL = "http://localhost:4000/v1/gpu/deploy"

# Read the training script content
script_path = Path(__file__).parent / "finetune_smart_turn_v3.py"
script_content = script_path.read_text()

# Escape for embedding in a Python string inside shell
# We'll write it from Python using a heredoc-style approach
script_lines = script_content.replace("\\", "\\\\").replace("'", "'\\''")

# Docker start command — writes script from shell heredoc, starts health server + training
docker_cmd = r"""bash -c '
set -ex
echo "[finetune] Starting at $(date)"

export PIP_CACHE_DIR=/workspace/.pip_cache
export HF_HOME=/workspace/huggingface
export TRANSFORMERS_CACHE=/workspace/huggingface
export TMPDIR=/workspace/tmp
mkdir -p $PIP_CACHE_DIR $HF_HOME $TMPDIR /workspace/checkpoints

# Health server + log tail on :8000 (gateway expects "status":"ok")
python3 << "HEALTH_EOF" &
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os, subprocess

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        ckpt = os.path.exists("/workspace/checkpoints/smart_turn_pt_v3/resume_checkpoint.pt")
        done = os.path.exists("/workspace/checkpoints/smart_turn_pt_v3/training_results.json")
        log_tail = ""
        if self.path == "/logs":
            try:
                log_tail = subprocess.check_output(["tail", "-100", "/workspace/training.log"], stderr=subprocess.DEVNULL).decode()
            except: pass
            self.wfile.write(log_tail.encode())
            return
        self.wfile.write(json.dumps({"status": "ok", "training": True, "has_checkpoint": ckpt, "done": done}).encode())
    def log_message(self, *a): pass

HTTPServer(("0.0.0.0", 8000), H).serve_forever()
HEALTH_EOF

echo "[finetune] Health server started on :8000"

# Install system deps (FFmpeg for audio decoding)
echo "[finetune] Installing system deps..."
apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 > /dev/null 2>&1
echo "[finetune] Installing Python deps..."
pip install --quiet 'transformers' 'datasets>=2.18,<3.0' torchaudio soundfile librosa 2>&1 | tail -5
echo "[finetune] Deps installed"

# Write training script (passed via TRAINING_SCRIPT env var, gzip+base64)
echo "[finetune] Decoding training script..."
echo "$TRAINING_SCRIPT_GZ" | base64 -d | gunzip > /workspace/finetune_smart_turn_v3.py
ls -la /workspace/finetune_smart_turn_v3.py

# Check for existing checkpoint
if [ -f /workspace/checkpoints/smart_turn_pt_v3/resume_checkpoint.pt ]; then
    echo "[finetune] Found resume checkpoint — continuing training"
fi

# Run training
echo "[finetune] Starting training..."
cd /workspace
python3 finetune_smart_turn_v3.py 2>&1 | tee /workspace/training.log

echo "[finetune] Training complete at $(date)"
sleep infinity
'"""

# Compress script with gzip before base64 to reduce size
import gzip
import base64
script_gz = gzip.compress(script_content.encode())
script_gz_b64 = base64.b64encode(script_gz).decode()

body = {
    "provider": "tensordock",
    "dockerImage": "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
    "gpuTypes": [
        "NVIDIA A40",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX A6000",
        "NVIDIA GeForce RTX 3090",
    ],
    "containerDiskInGb": 30,
    "dockerStartCmd": docker_cmd,
    "env": {
        "TRAINING_SCRIPT_GZ": script_gz_b64,
    },
}

print(f"Deploying fine-tuning pod...")
print(f"  Script size: {len(script_gz_b64)} bytes (gzip+base64, original {len(script_content)} bytes)")
print(f"  Docker image: {body['dockerImage']}")
print(f"  GPU types: {body['gpuTypes']}")
print(f"  Container disk: {body['containerDiskInGb']} GB")

req = urllib.request.Request(
    GATEWAY_URL,
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        print(f"\nDeploy response: {json.dumps(result, indent=2)}")
except urllib.error.HTTPError as e:
    error_body = e.read().decode()
    print(f"\nDeploy failed (HTTP {e.code}): {error_body}")
except Exception as e:
    print(f"\nDeploy failed: {e}")
