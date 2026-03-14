#!/bin/bash
set -e

echo "=== Turn-Taking Benchmark - Vast.ai Runner ==="
echo "Started at: $(date)"

cd /workspace

# Clone benchmark repo
if [ ! -d "turn-taking-study" ]; then
    git clone https://github.com/marcosremar/turn-taking-study.git
fi
cd turn-taking-study

# Install Python deps
pip install --no-cache-dir -r requirements.txt 2>&1 | tail -5

# Install VAP
if [ ! -d "/workspace/vap" ]; then
    echo "=== Cloning VAP ==="
    git clone https://github.com/ErikEkstedt/VoiceActivityProjection.git /workspace/vap
    cd /workspace/vap && pip install -e . 2>&1 | tail -5
    cd /workspace/turn-taking-study
fi

# Install VAP dataset tools
if [ ! -d "/workspace/vap_dataset" ]; then
    echo "=== Cloning VAP Dataset Tools ==="
    git clone https://github.com/ErikEkstedt/vap_dataset.git /workspace/vap_dataset
    cd /workspace/vap_dataset && pip install -e . 2>&1 | tail -5
    cd /workspace/turn-taking-study
fi

echo "=== Setup complete, running benchmarks ==="

# Run benchmarks
python run_benchmarks.py --all 2>&1 | tee /workspace/benchmark.log

echo "=== Generating report ==="
python generate_report.py 2>&1 | tee -a /workspace/benchmark.log

echo "=== DONE at $(date) ==="
echo "Results in /workspace/turn-taking-study/results/"
echo "Report in /workspace/turn-taking-study/report/"

# Keep instance alive for result collection
sleep infinity
