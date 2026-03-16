---
title: "Pipecat Smart Turn Data v3.2 - Training Dataset Card"
source: https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train
date: 2026-03-16
---

# Smart Turn Data v3.2 Training Dataset

The official training dataset for **Smart Turn v3.2**, hosted on Hugging Face Datasets.

## Key Specifications

| Attribute | Value |
|-----------|-------|
| **Organization** | Pipecat |
| **Dataset Size** | 100K - 1M rows |
| **Total Rows** | 270,946 |
| **Download Size** | 41.4 GB |
| **Parquet Size** | 41.4 GB |
| **Format** | Parquet |
| **Split** | train (271k rows) |
| **Subset** | default (271k rows) |

## Data Modalities

- **Audio** - Audio samples with duration ranging from 0.36s to 32.6s
- **Text** - Language and metadata fields

## Supported Libraries

- Datasets
- Dask
- Polars

## Dataset Schema

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `audio` | AudioObject | Audio samples |
| `id` | Text | UUID identifier (36 chars) |
| `language` | Text | 23 language classes |
| `endpoint_bool` | Boolean | End-of-turn indicator |
| `midfiller` | Boolean | Mid-utterance filler detection |
| `endfiller` | Boolean | End-of-utterance filler detection |
| `synthetic` | Boolean | Synthetic data flag |
| `dataset` | Text | Source dataset identifier (12 classes) |
| `spoken_text` | Null | Skipped column |

### Language Coverage

English, Chinese (zho), Vietnamese, Finnish, Spanish, Bengali, Hindi, Japanese, Portuguese, Russian, Korean, German, French, Dutch, Arabic, Marathi, Turkish, Icelandic, Polish, Ukrainian, Italian, Indonesian, Norwegian

### Dataset Sources

- chirp3_1, chirp3_2
- liva_1
- midcentury_1
- mundo_1
- rime_2
- orpheus_endfiller_1

## Metadata (Croissant Format)

The dataset uses **Croissant ML Commons** metadata schema (v1.1):

```json
{
  "@context": "https://schema.org/",
  "conformsTo": "http://mlcommons.org/croissant/1.1",
  "@type": "sc:Dataset",
  "name": "smart-turn-data-v3.2-train"
}
```

## Contributors

### Direct Contributors
- The Pipecat team
- [Liva AI](https://www.theliva.ai/)
- [Midcentury](https://www.midcentury.xyz/)
- [MundoAI](https://mundoai.world/)

### Background Noise Attribution
CC-0 licensed background noise samples sourced from Freesound.org contributors including:
- 4team, tomhannen, craigsmith, mrmayo, martats, and 26+ additional contributors

## Access Methods

- **Dataset Viewer:** https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train/viewer/
- **Data Studio:** https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train/viewer/default/train
- **Files Browser:** Git repository with parquet conversion available
