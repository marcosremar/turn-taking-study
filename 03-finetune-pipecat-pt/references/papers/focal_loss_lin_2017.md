---
title: "Focal Loss for Dense Object Detection"
authors:
  - Tsung-Yi Lin
  - Priya Goyal
  - Ross Girshick
  - Kaiming He
  - Piotr Dollar
year: 2017
source: https://arxiv.org/abs/1708.02002
date_converted: 2026-03-16
---

## Abstract

The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. The authors discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. They propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. The novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. Using focal loss, their one-stage RetinaNet detector matches the speed of previous one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors.

## Key Contributions

1. **Focal Loss function**: A dynamically scaled cross entropy loss where the scaling factor decays to zero as confidence in the correct class increases, automatically down-weighting easy examples and focusing on hard examples.
2. **RetinaNet**: A simple one-stage dense object detector built on Feature Pyramid Networks (FPN) that, when trained with focal loss, achieves state-of-the-art results.
3. **Identification of class imbalance** as the central obstacle preventing one-stage detectors from matching two-stage detector accuracy.

## Method Details

### Focal Loss Definition

Standard cross entropy loss:

```
CE(pt) = -log(pt)
```

Focal loss adds a modulating factor `(1 - pt)^gamma`:

```
FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
```

Key properties:
- When `gamma = 0`, FL is equivalent to CE.
- When an example is misclassified (`pt` is small), the modulating factor is near 1 and the loss is unaffected.
- As `pt -> 1`, the factor goes to 0, down-weighting well-classified examples.
- With `gamma = 2`, an example classified with `pt = 0.9` has 100x lower loss compared to CE; with `pt ~ 0.968`, 1000x lower loss.
- Best setting found: `gamma = 2`, `alpha = 0.25`.

### RetinaNet Architecture

- **Backbone**: ResNet + Feature Pyramid Network (FPN), pyramid levels P3-P7 with C=256 channels.
- **Classification subnet**: Small FCN with four 3x3 conv layers (256 filters, ReLU) + final 3x3 conv with KA filters + sigmoid. Shared across all pyramid levels.
- **Box regression subnet**: Identical structure to classification subnet, terminates in 4A linear outputs per location. Class-agnostic.
- **Anchors**: 9 anchors per level (3 aspect ratios x 3 scales), covering 32-813 pixel range.
- **Initialization**: Prior probability pi=0.01 for foreground class at initialization to prevent instability from background class dominance.

## Experimental Results

### Ablation Studies (ResNet-50-FPN, COCO minival)

| Loss | gamma | alpha | AP | AP50 | AP75 |
|------|-------|-------|------|------|------|
| CE (balanced) | 0 | 0.75 | 31.1 | 49.4 | 33.0 |
| FL | 0.5 | 0.75 | 31.4 | 49.9 | 33.1 |
| FL | 1.0 | 0.50 | 32.9 | 51.7 | 35.2 |
| **FL** | **2.0** | **0.25** | **34.0** | **52.5** | **36.5** |
| FL | 5.0 | 0.25 | 32.2 | 49.6 | 34.8 |

### FL vs. OHEM (ResNet-101-FPN)

| Method | AP | AP50 | AP75 |
|--------|------|------|------|
| OHEM (best) | 32.8 | 50.3 | 35.1 |
| FL | **36.0** | **54.9** | **38.7** |

FL outperforms the best OHEM variant by **3.2 AP points**.

### State-of-the-Art Comparison (COCO test-dev)

| Method | Backbone | AP | AP50 | AP75 |
|--------|----------|------|------|------|
| Faster R-CNN w FPN | ResNet-101-FPN | 36.2 | 59.1 | 39.0 |
| Faster R-CNN by G-RMI | Inception-ResNet-v2 | 34.7 | 55.5 | 36.7 |
| DSSD513 | ResNet-101-DSSD | 33.2 | 53.3 | 35.2 |
| **RetinaNet** | **ResNet-101-FPN** | **39.1** | **59.1** | **42.3** |
| **RetinaNet** | **ResNeXt-101-FPN** | **40.8** | **61.1** | **44.1** |

### Speed vs. Accuracy

- RetinaNet-101-600: 122ms inference, matching Faster R-CNN accuracy (36.0 AP) at similar speed.
- RetinaNet-101-800: 198ms inference, 37.8 AP -- surpassing all prior methods.

## Relevance to Turn-Taking / End-of-Turn Detection

Focal loss is directly applicable to end-of-turn detection for BabelCast:

1. **Class imbalance problem**: End-of-turn detection faces severe class imbalance -- the vast majority of audio frames are "continuing speech" (easy negatives), while actual turn-end boundaries are rare events. This is analogous to the foreground-background imbalance in object detection.

2. **Down-weighting easy examples**: In a streaming ETD system, most 100ms frames are clearly mid-utterance. Focal loss would prevent these trivially classified frames from dominating the gradient, focusing learning on the ambiguous boundary cases (pauses vs. turn-ends).

3. **Drop-in replacement**: Focal loss is a simple modification to standard cross entropy -- just adding `(1 - pt)^gamma` -- making it trivial to integrate into any binary or ternary classifier for end-of-turn detection (e.g., the GRU or Wav2Vec models in SpeculativeETD).

4. **Hyperparameter robustness**: The paper shows `gamma = 2, alpha = 0.25` works well across settings, reducing the need for extensive tuning.

5. **No sampling needed**: Unlike OHEM or hard negative mining, focal loss operates on all examples naturally, which is important for real-time streaming where we process every frame.
