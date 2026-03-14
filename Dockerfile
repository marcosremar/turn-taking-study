FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace/turn-taking-study

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone VAP repo
RUN git clone https://github.com/ErikEkstedt/VoiceActivityProjection.git /workspace/vap && \
    cd /workspace/vap && pip install -e .

# Clone VAP dataset tools
RUN git clone https://github.com/ErikEkstedt/vap_dataset.git /workspace/vap_dataset && \
    cd /workspace/vap_dataset && pip install -e .

# Copy benchmark scripts
COPY . .

# Download models on build (cache in image)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('livekit/turn-detector')"
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=True)"

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_CACHE=/workspace/hf_cache

CMD ["python", "run_benchmarks.py", "--all"]
