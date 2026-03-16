---
title: "Pipecat Smart Turn v3.2 - README"
source: https://raw.githubusercontent.com/pipecat-ai/smart-turn/main/README.md
date: 2026-03-16
---

# Smart Turn v3.2

An open-source, community-driven native audio turn detection model designed to improve upon traditional voice activity detection (VAD) approaches. The model determines when a voice agent should respond to human speech by analyzing linguistic and acoustic cues rather than simply detecting speech presence.

## Key Features

**Language Support:** The system supports 23 languages including Arabic, Bengali, Chinese, Danish, Dutch, German, English, Finnish, French, Hindi, Indonesian, Italian, Japanese, Korean, Marathi, Norwegian, Polish, Portuguese, Russian, Spanish, Turkish, Ukrainian, and Vietnamese.

**Performance Characteristics:**
- Inference completes in as little as 10ms on certain CPUs
- Most cloud instances see sub-100ms execution times
- Integrates efficiently with lightweight VAD models like Silero
- Two versions available: 8MB quantized CPU variant and 32MB unquantized GPU variant
- GPU version uses fp32 weights for marginally faster inference and ~1% accuracy improvement
- CPU version uses int8 quantization for reduced size and speed with minimal accuracy trade-off

**Technical Approach:** The model operates directly on PCM audio samples rather than text transcriptions, capturing prosodic nuances that inform turn-taking decisions.

## Setup Instructions

**Environment Configuration:**
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Platform-Specific Dependencies:**

Ubuntu/Debian systems require:
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev
```

macOS with Homebrew:
```bash
brew install portaudio
```

**Running the Demonstration:**
```bash
python record_and_predict.py
```

Initial startup requires approximately 30 seconds. Test phrases include "I can't seem to, um ..." and "I can't seem to, um, find the return label."

## Audio Input Specifications

The model accepts 16kHz mono PCM audio with maximum duration of 8 seconds. The recommended approach involves providing the complete audio of the current user turn. When audio exceeds 8 seconds, truncate from the beginning while maintaining context.

For shorter audio, prepend zero-value padding to reach the required length, ensuring actual speech content occupies the end of the input vector.

## Model Architecture

Smart Turn v3 employs Whisper Tiny as its foundation with an added linear classification layer. The transformer-based architecture contains approximately 8 million parameters. Development involved experimentation with wav2vec2-BERT, wav2vec2, LSTM implementations, and additional transformer classifier configurations.

## Integration Methods

**Pipecat Integration:** The framework supports local inference via `LocalSmartTurnAnalyzerV3` (version 0.0.85 and later). On Pipecat Cloud's standard 1x instance, inference typically completes in around 65ms.

**Direct Integration:** Import `model.py` and `inference.py` from the Smart Turn repository and invoke the `predict_endpoint()` function. Reference implementation available in `predict.py`.

## Training Infrastructure

Training code resides in `train.py` and downloads datasets from the pipecat-ai HuggingFace repository. Training can execute locally or via Modal using `train_modal.py`. Training sessions log to Weights & Biases unless disabled.

**Training command for Modal:**
```bash
modal run --detach train_modal.py
```

**Current Datasets:**
- pipecat-ai/smart-turn-data-v3.2-train
- pipecat-ai/smart-turn-data-v3.2-test

## Community Contributions

**Data Classification:** Manual training data categorization assistance needed at https://smart-turn-dataset.pipecat.ai/

**Human Data Contribution:** Participants can contribute through turn training games at https://turn-training.pipecat.ai/ or by following the data generation contribution guide.

**Future Development Areas:**
- Additional language support expansion
- Performance optimization and architecture refinement
- Expanded human dataset collection
- Text-conditioned inference for specialized input modes
- Training platform diversification

## Licensing and Attribution

Smart Turn operates under the BSD 2-clause license, permitting unrestricted usage, modification, and contribution.

**Project Contributors:**
- Marcus (marcus-daily)
- Eli (ebb351)
- Mark (markbackman)
- Kwindla (kwindla)

**Data Contributors:**
- Liva AI
- Midcentury
- MundoAI

## Resources

- [HuggingFace Model Repository](https://huggingface.co/pipecat-ai/smart-turn-v3)
- [Pipecat Documentation](https://docs.pipecat.ai/server/utilities/smart-turn/smart-turn-overview)
- [Pipecat Framework](https://pipecat.ai)
