---
title: "Pipecat Smart Turn - Data Generation Contribution Guide"
source: https://raw.githubusercontent.com/pipecat-ai/smart-turn/main/docs/data_generation_contribution_guide.md
date: 2026-03-16
---

# Contributing Training Data to Smart Turn

## Audio Format Requirements

**FLAC is the preferred format** for training data, with lossy formats like MP3 or Opus discouraged. Audio should be **mono audio with a bit depth of 16 bits** and works optimally at 16kHz sample rates, though higher rates are acceptable.

## File Organization

Contributors have flexibility in naming and directory structure. The recommended approach uses unique identifiers for files combined with directory labels by language and completeness status (e.g., `eng/incomplete/b3799254-8d6c-11f0-a90e-e7e92780240b.flac`).

## Audio Length and Variation

Each file must contain **one speech sample, no longer than 16 seconds.** Variety in length is encouraged, ranging from single words to complex sentences.

## Content Guidelines

Samples should represent **a single turn in the conversation** with only one speaker per file. Speech should resemble interactions with voice assistants or customer service representatives. The documentation emphasizes avoiding sentence repetition and background noise while excluding real personally identifiable information.

## Complete vs. Incomplete Classification

Samples require binary labeling. "Complete" samples represent finished thoughts suitable for immediate response, while "Incomplete" samples suggest the speaker will continue, ending with filler words, connectives, or suggestive prosody. Critically, incomplete samples **must not be cut off in the middle of a word** but rather end with full words and approximately 200ms of silence.

A 50:50 split between categories is recommended for unbiased training.

## Licensing and Submission

Contributors must own recordings and secure speaker consent for public release. Datasets are published via HuggingFace. Submissions can occur through various cloud storage methods, with GitHub issues as the contact point.
