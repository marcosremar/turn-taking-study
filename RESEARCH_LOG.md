# Smart Turn Portuguese Fine-Tuning — Research Log

## Objetivo

Fine-tuning do modelo Pipecat Smart Turn (detecção de fim de turno em conversas) para português brasileiro, visando melhorar a acurácia de ~68% (modelo original inglês aplicado a PT) para 90%+.

## Background

### O que é Smart Turn
- Modelo do framework [Pipecat](https://github.com/pipecat-ai/smart-turn) para detectar se um falante terminou de falar
- Arquitetura: Whisper encoder + attention pooling + classificador binário (complete/incomplete)
- Janela de 8 segundos de áudio, saída binária: "turno completo" vs "turno incompleto"
- Modelo original treinado em inglês (v3.1), 39M parâmetros (Whisper Tiny)

### Por que fine-tuning em português
- O modelo original em inglês tem ~68.6% de acurácia em português
- Prosódia, entonação e padrões de turn-taking são diferentes entre idiomas
- Aplicação: tradução em tempo real de reuniões (BabelCast)

---

## Fase 1: Benchmark do modelo original (pré fine-tuning)

### Datasets de avaliação
- **NURC-SP Corpus Minimo** (nilc-nlp): diálogos reais em PT-BR espontâneo dos anos 1970-1990
- Scripts: `setup_nurc_dataset.py`, `benchmark_pipecat.py`

### Resultado baseline
- Acurácia do Smart Turn v3.1 em português: **~68.6%**
- Problema principal: modelo não entende padrões prosódicos do português

---

## Fase 2: Fine-tuning v2 — Heurística de corte (2026-03-14)

### Abordagem
- Baixar datasets de ASR em português do HuggingFace
- Criar labels heurísticas: "complete" = final da frase, "incomplete" = corte aleatório a 30-75%
- Treinar com speaker-based split para evitar data leakage

### Datasets utilizados
| Dataset | Tipo | Horas | Samples |
|---------|------|-------|---------|
| CORAA v1.1 | Conversacional BR-PT | 291h | 7,000 |
| MLS Portuguese | Audiobook (leitura) | 168h | 7,000 |
| CORAA-MUPE-ASR | Entrevistas | 365h | 7,000 |
| **Total** | | | **21,000** |

### Configuração de treino
- **Modelo**: Whisper Tiny encoder (39M params)
- **GPU**: RTX 3090 24GB (Vast.ai, $0.069/hr)
- **Batch size**: 32
- **Learning rate**: 2e-5 (encoder: 2e-6, head: 2e-5)
- **Augmentation**: volume scaling, Gaussian noise
- **Split**: 13,032 train (4,100 speakers) / 698 val (512 speakers) / 7,270 test (512 speakers)
- **Early stopping**: patience=5 no val_f1

### Infraestrutura (problemas e soluções)
- **6+ instâncias Vast.ai morreram** durante o treino (spot instances instáveis)
- **3 pods RunPod falharam** (PyTorch images too large for container disk)
- **Solução**: on-demand Vast.ai instance, RTX 3090, reliability=1.00
- **Mac local**: treinamento funciona mas deixa a máquina muito lenta (MPS)
- **Dependências críticas**: PyTorch >= 2.4, `datasets<4` (para evitar torchcodec), librosa, typing_extensions >= 4.12

### Resultados v2

#### Progresso por época
| Epoch | Train Loss | Train Acc | Val Acc | Val F1 | Val Prec | Val Rec |
|-------|-----------|-----------|---------|--------|----------|---------|
| 1 | 0.6032 | 0.668 | 0.669 | 0.721 | 0.600 | 0.903 |
| 2 | 0.5019 | 0.748 | 0.716 | 0.751 | 0.643 | 0.903 |
| 3 | 0.4521 | 0.782 | 0.716 | 0.751 | 0.643 | 0.903 |
| 4 | 0.4161 | 0.806 | 0.749 | 0.748 | 0.714 | 0.785 |
| 5 | 0.3962 | 0.817 | 0.742 | 0.717 | 0.748 | 0.689 |
| **6** | **0.3806** | **0.831** | **0.754** | **0.760** | **0.707** | **0.822** |
| 7 | 0.3703 | 0.832 | 0.742 | 0.748 | 0.697 | 0.807 |
| 8 | 0.3566 | 0.845 | 0.739 | 0.709 | 0.752 | 0.671 |
| 9 | 0.3449 | 0.846 | 0.748 | 0.745 | 0.716 | 0.776 |
| 10 | 0.3390 | 0.853 | 0.748 | 0.758 | 0.695 | 0.834 |
| 11 | 0.3237 | 0.865 | 0.754 | 0.756 | 0.713 | 0.804 |

Early stopping na época 11 (sem melhora por 5 épocas).

#### Melhor modelo (época 6)
- **Val Accuracy**: 75.4%
- **Val F1**: 0.760
- **Val Precision**: 70.7%
- **Val Recall**: 82.2%

#### Teste (speakers totalmente novos)
- **Test Accuracy**: 64.4%
- **Test F1**: 0.600
- **Test Precision**: 68.0%
- **Test Recall**: 53.7%
- TP=1945, FP=916, FN=1674, TN=2735

### Análise dos problemas da v2

1. **Labels heurísticas (problema principal)**: Cortar frases aleatoriamente em 30-75% não simula turn-taking real. O modelo aprendeu "tem silêncio no final?" em vez de "a pessoa terminou de falar?"

2. **MLS é audiobook**: 1/3 dos dados são leitura de audiobook — prosódia completamente diferente de conversação real

3. **MUPE speaker_id quebrado**: Usava `speaker_type` ("interviewer"/"interviewee") como speaker_id, comprometendo o speaker split

4. **Modelo pequeno**: Whisper Tiny (39M params) tem capacidade limitada para capturar padrões prosódicos complexos

5. **Gap val/test grande (75.4% vs 64.4%)**: Indica que o modelo não generaliza bem para speakers novos

### Arquivos gerados
- `checkpoints/smart_turn_pt_v2/best_model.pt` — 31MB PyTorch checkpoint
- `checkpoints/smart_turn_pt_v2/smart_turn_pt.onnx` + `.onnx.data` — 31MB ONNX
- `checkpoints/smart_turn_pt_v2/finetune.log` — log completo

### Commits
- `4517458` — feat: Pipecat Smart Turn Portuguese evaluation, fine-tuning pipeline
- `307b0fd` — feat: GPU fine-tuning script with HuggingFace Portuguese datasets
- `62e1816` — fix: total_mem → total_memory for PyTorch 2.10 compat
- `0039c15` — fix: reduce samples to 5k/dataset and workers to prevent OOM
- `51025dd` — fix: MPS support, skip Common Voice, increase samples to 7k/dataset

---

## Fase 3: Fine-tuning v3 — Labels por pontuação + Whisper Base (em andamento)

### Melhorias implementadas

1. **Labels baseadas em pontuação do texto**
   - Frase termina com `.` `!` `?` `…` → complete (1.0)
   - Frase termina com `,` `;` `:` `-` → incomplete (0.0)
   - Texto sem pontuação com ≤2 palavras → descartado (ambíguo)
   - Texto sem pontuação com 3+ palavras → incomplete (transcritor teria colocado ponto se fosse completa)

2. **Removido MLS audiobook** — só dados conversacionais (CORAA + MUPE)

3. **Whisper Base** (74M params) em vez de Whisper Tiny (39M)
   - 2x mais parâmetros no encoder
   - hidden_size: 512 (vs 384 no Tiny)
   - Melhor capacidade para capturar prosódia

4. **Speaker ID do MUPE corrigido** — usa hash do audio_path ou agrupamento por index

5. **Mais dados**: 25k samples por dataset (vs 7k na v2) = ~50k total

6. **Augmentation melhorada**:
   - Speed perturbation (0.9x–1.1x)
   - Volume scaling (0.6x–1.4x)
   - Gaussian noise mais agressivo
   - Time shift aleatório (±0.3s)

7. **LR schedule com warmup**: 2 épocas de warmup + cosine decay

8. **Classifier head maior**: 512→128→1 (vs 256→64→1)

9. **Patience aumentado**: 7 épocas (vs 5)

### Configuração
- **Modelo**: Whisper Base (74M params)
- **Datasets**: CORAA + MUPE (~50k samples)
- **LR**: 3e-5 (encoder: 3e-6)
- **Epochs**: até 30 (com early stopping patience=7)
- **Batch size**: 32

### Resultados v3
*(a ser preenchido após o treino)*

---

## Roadmap futuro (se necessário)

### Nível 4 — Dados reais de turn-taking
- Usar NURC-SP com anotações reais de fronteiras de turno
- Gravar dados de reuniões reais em português
- Combinar features de áudio + texto (multimodal)

### Nível 5 — Arquitetura avançada
- Whisper Small (244M params)
- Adicionar features linguísticas (completude sintática via LLM)
- Ensemble de modelos

### Limites teóricos
- Humanos discordam em ~10-15% dos casos de turn-taking
- Teto realista: 90-95%
- Algumas frases são genuinamente ambíguas ("Sim...", "É...")

---

## Referências

- Pipecat Smart Turn: https://github.com/pipecat-ai/smart-turn
- CORAA v1.1: https://huggingface.co/datasets/Racoci/CORAA-v1.1
- CORAA-MUPE-ASR: https://huggingface.co/datasets/nilc-nlp/CORAA-MUPE-ASR
- NURC-SP Corpus Minimo: https://huggingface.co/datasets/nilc-nlp/NURC-SP_Corpus_Minimo
- MLS Portuguese: https://huggingface.co/datasets/facebook/multilingual_librispeech
