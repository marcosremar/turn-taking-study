---
title: "Calibrating Imbalanced Classifiers with Focal Loss: An Empirical Study"
source: https://aclanthology.org/2022.emnlp-industry.14/
date: 2026-03-16
---

# Calibrating Imbalanced Classifiers with Focal Loss: An Empirical Study

**Authors:** Cheng Wang, Jorge Balazs, Gyuri Szarvas, Patrick Ernst, Lahari Poddar, Pavel Danchenko (Amazon)

**Venue:** EMNLP 2022 Industry Track, December 9-11, Abu Dhabi, UAE, pages 155-163

**DOI:** 10.18653/v1/2022.emnlp-industry.14

**PDF:** https://aclanthology.org/2022.emnlp-industry.14.pdf

## Abstract

Imbalanced data distribution is a practical and common challenge in building ML models in industry, where data usually exhibits long-tail distributions. For instance, in virtual AI Assistants (Google Assistant, Amazon Alexa, Apple Siri), the "play music" or "set timer" utterance is exposed to an order of magnitude more traffic than other skills. This can easily cause trained models to overfit to the majority classes, leading to model miscalibration. The uncalibrated models output unreliable (mostly overconfident) predictions, which are at high risk of affecting downstream decision-making systems. The authors empirically show the effectiveness of model training with focal loss in learning better calibrated models, as compared to standard cross-entropy loss. Better calibration enables better control of the precision-recall trade-off for trained models.

## 1. Introduction

Building ML models in industry faces practical challenges from imbalanced data distributions, particularly long-tail distributions, which make models overfit to majority data classes and lead to miscalibration -- the model-predicted probability fails to estimate the likelihood of true correctness and provides over- or under-confident predictions.

The study focuses on **return reason code prediction** in customer service chatbots as a practical application of focal loss for calibration.

### Contributions

- Empirically examine the effectiveness of using focal loss in handling model miscalibration in a practical application setting
- Show that good calibration is important to achieve a desired precision or recall target by tuning classification thresholds (standard cross-entropy loss is incapable of this due to skewed predicted probability distribution)
- Demonstrate performance of calibrated models through a chatbot serving customers across three conversational bot use-cases

## 2. Focal Loss Formulation

Focal loss is defined as:

```
L_f = - sum_{i=1}^{N} (1 - p_{i,y_i})^gamma * log(p_{i,y_i})
```

where `p_{i,y_i}` is predicted probability of the i-th sample and `gamma` is a hyper-parameter typically set to `gamma = 1`.

### Theoretical Interpretation

Focal loss can be interpreted as a trade-off between minimizing KL divergence and maximizing entropy:

```
L_f >= KL(q || p) + H(q) - gamma * H(p)
```

The rationale: we learn a probability `p` to have a high value (confident) due to the KL term, but not too high (overconfident) due to the entropy regularization term.

### Practical Advantages

Compared to other calibration methods (temperature scaling, Bayesian methods, label smoothing, kernel-based methods), focal loss:
- Neither increases computational overhead nor requires architectural modifications
- Offers **in-training implicit calibration** (unlike temperature scaling which requires post-training calibration)

## 3. Calibration Metrics

### Reliability Diagrams

Predictions are grouped into N interval bins; accuracy vs. confidence is computed per bin:

```
acc(b_n) = (1/I_n) * sum_i 1(y_hat_i = y_i)
conf(b_n) = (1/I_n) * sum_i p_hat_i
```

A perfectly calibrated model has `acc(b_n) = conf(b_n)` for all n.

### Expected Calibration Error (ECE)

```
ECE = (1/I) * sum_{n=1}^{N} I_n * |acc(b_n) - conf(b_n)|
```

### Maximum Calibration Error (MCE)

```
MCE = max_n |acc(b_n) - conf(b_n)|
```

Particularly important in high-risk applications. For a perfectly calibrated classifier, both ECE and MCE equal 0.

## 4. Datasets and Implementation

### Task

Binary and multi-class return reason code prediction in an online retail store:
- **Binary:** "item is defective or does not work" (Label 0) vs. OTHERS (Label 1)
- **Multi-class:** 4 specific return reasons + OTHERS (5 total classes)

Both datasets exhibit class imbalance with OTHERS as the most frequent class.

### Model Architecture

- 2 bidirectional LSTM layers + 2 dense layers
- Embedding dimension: 1024
- Hidden layer dimension: 128 (binary), 512 (multi-class)
- Dropout: 0.1 (embedding), 0.2 (dense)
- Optimizer: Adam
- Framework: PyTorch

### Golden Dataset

- Binary model: 1,013 human-annotated samples
- Multi-class model: 1,839 human-annotated samples
- Data split: 8:1:1 (train/val/test)

## 5. Results

### Binary Reason Code Prediction (Table 1)

| Metric | CE | FL1 | FL3 | FL5 | FL10 |
|--------|------|------|------|------|------|
| Accuracy | **0.836** | 0.824 | 0.831 | 0.834 | 0.816 |
| Precision | **0.838** | 0.822 | 0.830 | 0.834 | 0.807 |
| Recall | **0.823** | 0.814 | 0.821 | 0.823 | 0.805 |
| F1 | **0.828** | 0.817 | 0.824 | 0.827 | 0.806 |
| NLL | 2.159 | 1.438 | 0.608 | 0.258 | **0.178** |
| ECE | 0.168 | 0.166 | 0.139 | 0.080 | **0.078** |
| MCE | 0.720 | 0.730 | 0.236 | **0.134** | 0.143 |

**Key finding:** CE achieves best predictive performance, but FL significantly outperforms on calibration metrics (NLL, ECE, MCE). Higher gamma yields better calibration with modest predictive performance loss.

### Multi-Reason Code Prediction (Table 2)

| Metric | CE | FL1 | FL5 |
|--------|------|------|------|
| Accuracy | 0.751 | **0.760** | 0.751 |
| Precision | **0.814** | 0.807 | **0.814** |
| Recall | **0.757** | 0.755 | **0.757** |
| F1 | **0.764** | 0.760 | **0.764** |
| NLL | 0.599 | 0.429 | **0.309** |
| ECE | 0.037 | **0.023** | 0.037 |
| MCE | 0.296 | **0.197** | 0.299 |

### Reliability Diagrams

- CE model: ECE = 16.78%, MCE = 72.00% (binary)
- FL5 model: ECE = 7.98%, MCE = 13.40% (binary)
- FL10 model: ECE = 7.76%, MCE = 14.34% (binary)

As gamma increases from CE to FL10, probability distributions shift from "spiking" (overconfident, p close to 0 or 1) to flatter distributions (e.g., p = {0.6, 0.4}).

### Precision-Recall Trade-Off

CE model produces polarized probabilities, making it difficult to tune precision based on a given recall or vice versa. FL models learn better-distributed probabilities across [0, 1], enabling effective threshold-based precision/recall tuning.

## 6. Deployment Results (Online A/B Test)

Model deployed in three conversational chatbot use-cases with threshold=0.512 for 85% target precision:

### Online Evaluation (Table 3) -- Relative Improvements

| Application | Metric | Treatment vs. Control |
|------------|--------|----------------------|
| Use-case A | AR | +2.13% |
| Use-case A | PRR | +3.18% |
| Use-case A | 24RR | -0.65% |
| Use-case B | AR | +2.10% |
| Use-case B | PRR | +0.97% |
| Use-case B | 24RR | -0.68% |
| Use-case C | AR | +3.98% |
| Use-case C | PRR | +12.85% |
| Use-case C | 24RR | -1.02% |

**Metrics:**
- **AR (Automation Rate):** % contacts resolved without human involvement (higher = better)
- **PRR (Positive Response Rate):** % positive customer responses to chatbot resolution (higher = better)
- **24RR (Repeat Rate):** % customers contacting again within 24h for same issue (lower = better)

### Intrinsic Evaluation

- Precision on deployed model: 384/485 = 83.8% (aligns with offline 81.4%)
- Negative predictive value: 194/200 = 97% for OTHERS class

## 7. Key Takeaways

1. Focal loss provides simple, effective in-training calibration via entropy regularization
2. Higher gamma values improve calibration without significantly hurting predictive performance
3. Well-calibrated models enable practical precision/recall threshold tuning that CE-trained models cannot achieve
4. The discrimination-calibration trade-off is modest -- best model balances both (FL5 recommended)
5. Better calibration directly translates to improved downstream application metrics

## Citation

```bibtex
@inproceedings{wang-etal-2022-calibrating,
    title = "Calibrating Imbalanced Classifiers with Focal Loss: An Empirical Study",
    author = {Wang, Cheng and Balazs, Jorge and Szarvas, Gy{\"o}rgy and Ernst, Patrick and Poddar, Lahari and Danchenko, Pavel},
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    pages = "155--163",
    doi = "10.18653/v1/2022.emnlp-industry.14"
}
```

## References (Selected)

- Lin et al., 2017 -- Focal loss for dense object detection (original focal loss paper)
- Mukhoti et al., 2020 -- Calibrating deep neural networks using focal loss (NeurIPS)
- Guo et al., 2017 -- On calibration of modern neural networks (ICML)
- Pereyra et al., 2017 -- Regularizing neural networks by penalizing confident output distributions
- Naeini et al., 2015 -- Obtaining well calibrated probabilities using Bayesian binning (AAAI)
