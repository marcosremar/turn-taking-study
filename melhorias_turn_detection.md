# Smart Turn v3 — Deteccao de Fim de Turno para Portugues

## O que e

Modelo de deteccao de fim de turno (end-of-turn detection) para o BabelCast. Analisa os ultimos 8 segundos de audio e decide se o falante terminou de falar ou se esta apenas pausando. Isso evita que a traducao comece antes da hora (interrompendo o falante).

O modelo usa o **encoder do Whisper Tiny** (pre-treinado em 680.000 horas de audio multilingual) como extrator de features acusticas — entonacao, ritmo, energia, padroes espectrais — seguido de attention pooling + classificador MLP. **Nao usa o Whisper pra transcrever** — apenas como backbone de audio.

Arquitetura identica ao [Pipecat Smart Turn v3](https://github.com/pipecat-ai/smart-turn) original (Daily.co).

## Como funciona

```
Silero VAD detecta 200ms de silencio
         |
Smart Turn recebe os ultimos 8s de audio (16kHz mono)
         |
Whisper Feature Extractor → mel-spectrogram (80 bins x 800 frames)
         |
Whisper Tiny encoder (39M params) → representacoes acusticas (384-dim x 400 frames)
         |
Attention Pooling → aprende QUAIS frames sao importantes pra decisao
         |
Classifier MLP (384→256→64→1) → sigmoid → probabilidade [0, 1]
         |
Se probabilidade > threshold → "Turno completo" (pode comecar a traduzir)
Se probabilidade <= threshold → "Ainda falando" (espera mais)
```

O **attention pooling** foca nos frames perto do silencio, onde a entonacao final e a queda de energia sao mais informativas. O encoder captura:

- **Prosodia / Entonacao** — pitch caindo = fim de frase; pitch suspenso = pausa de hesitacao
- **Ritmo / Velocidade** — desaceleracao indica fim de pensamento
- **Energia / Volume** — queda de energia no final vs. manutencao na hesitacao
- **Padroes espectrais** — respiracao, fillers ("hum", "eh"), tipo de silencio

Por isso e muito melhor que VAD simples, que so detecta silencio.

## Dados de Treino

- **CORAA v1.1** — 291h de portugues brasileiro conversacional (HuggingFace: `Racoci/CORAA-v1.1`)
- **CORAA-MUPE-ASR** — 365h de entrevistas (HuggingFace: `nilc-nlp/CORAA-MUPE-ASR`)
- **15.000 amostras** (7.500 por dataset), 5.590 falantes
- **Labels hibridos**: pontuacao do texto (.!? = completo, ,;: = incompleto) + corte de audio em 30-75% pra amostras sem pontuacao
- Split por falante (train/val/test) pra evitar data leakage

## Historico de Treinamento

### Rodada 1 — Whisper Base + BCE Loss (baseline)

Primeiro experimento, usando encoder maior (Whisper Base, 74M params, hidden 512).

| Metrica | Teste |
|---------|-------|
| **F1** | 0.796 |
| Accuracy | 79.4% |
| Precision | 75.8% |
| Recall | 83.9% |
| Modelo (PT) | 78.2 MB |
| Best epoch | 12/30 (early stop 19) |

Treinado em Modal (A10 GPU), ~29 minutos.

### Rodada 2 — Whisper Tiny + BCE Loss

Trocamos pra Whisper Tiny (39M params, hidden 384) — mesmo backbone do Pipecat original. Modelo 2.5x menor, inferencia mais rapida, qualidade praticamente igual.

| Metrica | Teste |
|---------|-------|
| **F1** | 0.788 |
| Accuracy | 78.0% |
| Precision | 73.3% |
| Recall | 85.3% |
| Modelo (PT) | 30.5 MB |
| Best epoch | 13/30 (early stop 20) |

Treinado em Modal (A10 GPU), ~15 minutos.

**Conclusao**: diferenca de apenas 0.8% no F1 vs Whisper Base, com modelo 2.5x menor. Recall ate melhorou (+1.4%). Trade-off excelente.

### Rodada 3 — Whisper Tiny + Focal Loss + Label Smoothing (atual)

Aplicamos tres melhorias de precisao baseadas em pesquisa (ver secao "Solucoes Pesquisadas"):

1. **Focal Loss** (gamma=2.0, alpha=0.6) — penaliza falsos positivos, foca nos casos dificeis perto da fronteira de decisao
2. **Label Smoothing** (0.05) — labels viram 0.05/0.95 em vez de 0/1, melhora calibracao do modelo
3. **Threshold sweep** — avalia multiplos thresholds pra encontrar o melhor trade-off precisao/recall

| Metrica (threshold=0.5) | Teste |
|---------|-------|
| **F1** | **0.798** |
| Accuracy | 78.2% |
| Precision | 72.0% |
| Recall | **89.5%** |
| Modelo (PT) | 30.5 MB (~8 MB ONNX INT8) |
| Best epoch | 10/30 (early stop 17) |

**Threshold Sweep** — o grande ganho:

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.50 | 72.0% | 89.5% | 0.798 |
| 0.55 | 74.4% | 83.4% | 0.786 |
| **0.60** | **79.2%** | **75.4%** | **0.772** |
| **0.65** | **83.0%** | **64.9%** | **0.728** |
| 0.70 | 87.3% | 51.8% | 0.651 |
| 0.75 | 93.0% | 35.8% | 0.517 |
| 0.80 | 93.5% | 17.7% | 0.298 |

**Conclusao**: Focal Loss + Label Smoothing criaram um modelo muito mais calibrado. A precision sobe de 72% a 93% ajustando so o threshold. O sweet spot pra traducao simultanea e **threshold=0.60-0.65** (79-83% precision com recall razoavel).

### Comparativo das 3 rodadas

| Versao | Encoder | Loss | Threshold | Precision | Recall | F1 | Tamanho |
|--------|---------|------|-----------|-----------|--------|-----|---------|
| R1 | Whisper Base | BCE | 0.5 | 75.8% | 83.9% | 0.796 | 78.2 MB |
| R2 | Whisper Tiny | BCE | 0.5 | 73.3% | 85.3% | 0.788 | 30.5 MB |
| R3 | Whisper Tiny | Focal | 0.5 | 72.0% | 89.5% | **0.798** | 30.5 MB |
| **R3** | **Whisper Tiny** | **Focal** | **0.60** | **79.2%** | **75.4%** | **0.772** | **30.5 MB** |
| **R3** | **Whisper Tiny** | **Focal** | **0.65** | **83.0%** | **64.9%** | **0.728** | **30.5 MB** |

**Melhor resultado geral**: R3 com threshold=0.5 (F1 0.798) ou threshold=0.60 (precision 79.2%).

## Arquivos

```
docs/turn-taking-study/
  finetune_smart_turn_v3.py      # Script de treino (Whisper Tiny + Focal Loss)
  modal_finetune.py              # Deploy no Modal (A10G GPU, 4h timeout)
  deploy_finetune.py             # Deploy alternativo via ai-gateway (TensorDock/Vast)
  results/                       # Rodada 1 (Whisper Base)
    best_model.pt                # 78.2 MB
    training_results.json
  results-tiny/                  # Rodada 2 (Whisper Tiny + BCE)
    best_model.pt                # 30.5 MB
    training_results.json
  results-focal/                 # Rodada 3 (Whisper Tiny + Focal Loss) ← ATUAL
    best_model.pt                # 30.5 MB
    training_results.json
  melhorias_turn_detection.md    # Este documento
```

---

## Solucoes Pesquisadas para Melhorar Precisao

### Melhoria 1: Threshold de Confianca + Buffer de Confirmacao

**STATUS: IMPLEMENTADO (Rodada 3)**

O modelo retorna uma probabilidade (sigmoid). Em vez do threshold fixo de 0.5, usamos threshold configuravel. O Pipecat original usa 0.7 como default.

**Quem faz isso**: Pipecat (threshold 0.7), Krisp (threshold configuravel, 6% FPR com 0.9s mean shift time), AssemblyAI (dual detection com `end_of_turn_confidence_threshold`).

**Resultado**: Com threshold=0.65, precision sobe de 72% pra 83% (+11 pontos percentuais).

**Referencia**:
- [Krisp: Audio-only 6M Turn-Taking Model](https://krisp.ai/blog/turn-taking-for-voice-ai/)
- [AssemblyAI: Turn Detection](https://www.assemblyai.com/blog/turn-detection-endpointing-voice-agent)

---

### Melhoria 2: Focal Loss + Label Smoothing

**STATUS: IMPLEMENTADO (Rodada 3)**

- **Focal Loss** (gamma=2.0, alpha=0.6): penaliza exemplos faceis e foca nos casos dificeis perto da fronteira de decisao. Alpha < 1 penaliza mais os falsos positivos (dizer "terminou" quando nao terminou).
- **Label Smoothing** (0.05): labels viram 0.05/0.95 em vez de 0/1, evitando overconfidence e melhorando calibracao. Isso torna o threshold sweep muito mais eficaz.

**Quem faz isso**: Focal Loss (Lin et al. 2017, RetinaNet), calibracao com Focal Loss (EMNLP 2022), Asymmetric Loss (Ridnik et al. 2021, Alibaba DAMO).

**Resultado**: Recall subiu de 85.3% pra 89.5% no threshold=0.5, e o modelo ficou muito mais calibrado — precision controlavel de 72% a 93% via threshold.

**Referencia**:
- [Focal Loss for Dense Object Detection (Lin et al. 2017)](https://arxiv.org/abs/1708.02002)
- [Calibrating Imbalanced Classifiers with Focal Loss (EMNLP 2022)](https://aclanthology.org/2022.emnlp-industry.14/)

---

### Melhoria 3: Adicionar Texto (Multimodal — Audio + STT)

**STATUS: PENDENTE — proximo passo recomendado**

O modelo atual so ve audio. Adicionar a transcricao do STT como input adicional. Uma frase como "e depois eu..." e claramente incompleta — o texto da essa informacao mesmo quando o audio "parece" uma pausa natural.

Duas abordagens possiveis:

**a) Modelo separado de texto (pipeline):** Um LLM pequeno analisa a transcricao e da um score de completude. Combina com o score do modelo de audio por ensemble.

**b) Modelo unico multimodal (tipo Vogent):** Encoder de audio + texto com cross-attention, treinados juntos.

**Quem faz isso**:

- **LiveKit** — Qwen2.5-0.5B-Instruct, distillation de 7B → 0.5B. Resultado: **-39% falsos positivos** (interrupcoes). Funciona especialmente bem para entradas estruturadas (numeros, enderecos).
- **Vogent Turn 80M** (YC, 2025) — Whisper encoder (audio) + SmolLM2 ablated (texto), **94.1% accuracy** em ~7ms no T4. Estado da arte.
- **Speechmatics** — Semantic turn detection combinando features acusticas + linguisticas.

**Impacto esperado**: +5-10% F1. Para o BabelCast, a abordagem (a) e mais pratica: ja temos STT rodando (Whisper/Groq).

**Referencia**:
- [LiveKit: Improved End-of-Turn Model](https://blog.livekit.io/improved-end-of-turn-model-cuts-voice-ai-interruptions-39/)
- [Vogent Turn 80M](https://huggingface.co/vogent/Vogent-Turn-80M)
- [Speechmatics: Smarter Turn Detection](https://blog.speechmatics.com/semantic-turn-detection)

---

### Melhoria 4: Hard Negatives Sinteticos

**STATUS: PENDENTE**

Gerar amostras sinteticas dos padroes que mais causam falsos positivos:
- Pausas de hesitacao ("hum...", "eh...", "tipo...")
- Enumeracoes ("primeiro... segundo...")
- Pausas longas pra pensar (2-3 segundos) no meio de frase
- Frases sintaticamente completas mas semanticamente incompletas ("Eu fui la. E depois...")

**Quem faz isso**:

- **SpeculativeETD** (Samsung Research, arXiv 2503.23439) — criou dados sinteticos injetando fillers e estendendo hesitacoes
- **Vogent Turn 80M** — gerou "multi-clause responses, disfluent speech with filled pauses, list-like enumerations"
- **Deepgram** — 1.000 amostras hard curadas manualmente → 92%+ accuracy

**Impacto esperado**: +5-15% precision. Maior impacto potencial entre as melhorias pendentes.

**Referencia**:
- [SpeculativeETD (arXiv 2025)](https://arxiv.org/abs/2503.23439)
- [Deepgram: Evaluating End-of-Turn Detection](https://deepgram.com/learn/evaluating-end-of-turn-detection-models)

---

### Melhoria 5: Labels Reais de Turn-Taking (Melhor Dataset)

**STATUS: PENDENTE**

O maior limitante e como os labels sao criados:
- **Pontuacao do texto** — funciona, mas muitas amostras nao tem texto
- **Corte artificial em 30-75%** — o modelo aprende a detectar cortes bruscos, nao pausas reais de meio-frase

A melhoria e usar dados com **anotacao real de turn-taking**: onde um humano marcou onde cada turno comeca e termina.

**Quem faz isso**:

- **SpeculativeETD** — primeiro dataset publico para ETD: 122.481 amostras, 200+ horas. Anotacao ternaria: Speaking Unit, Pause, Gap.
- **Pipecat Smart Turn v3.1** — datasets publicados no HuggingFace com labels manuais shift/hold (270K amostras, 41GB)
- **VAP** (Erik Ekstedt) — aprendizado auto-supervisionado em conversas reais (Switchboard, Fisher)

**Impacto esperado**: +10% F1 ou mais. Maior investimento (coleta/anotacao de dados).

**Referencia**:
- [SpeculativeETD Dataset (arXiv 2025)](https://arxiv.org/abs/2503.23439)
- [Pipecat Smart Turn v3.1 Data (HuggingFace)](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.1-train)

---

### Melhoria 6: Contexto Conversacional (Historico de Turnos)

**STATUS: PENDENTE**

O modelo ve 8 segundos isolados. Mas turn-taking depende do contexto: se alguem fez uma pergunta, a resposta provavelmente sera longa.

**Quem faz isso**:

- **VAP** (Voice Activity Projection) — cross-attention transformer processando os dois canais de audio simultaneamente
- **LiveKit** — modelo Qwen2.5 recebe historico da conversa + frase atual

**Impacto esperado**: +5% F1. Mudanca arquitetural significativa.

**Referencia**:
- [VAP: Real-time Turn-taking Prediction (arXiv 2024)](https://arxiv.org/abs/2401.04868)
- [Multi-TPC Dataset (Nature 2026)](https://www.nature.com/articles/s41597-026-06819-x)

---

### Melhoria 7: Knowledge Distillation

**STATUS: PENDENTE**

Treinar um modelo grande (Whisper Small/Medium encoder + transformer classificador) como professor, depois destilar o conhecimento pro Whisper Tiny (aluno). O aluno aprende as soft probabilities do professor, capturando nuances dos casos dificeis.

**Quem faz isso**:

- **LiveKit** — Qwen 7B (professor) → Qwen 0.5B (aluno). Aluno "approaches teacher-level accuracy" com 14x menos params. Convergiu em ~1.500 steps.

**Impacto esperado**: +3-7% precision sobre treino direto do modelo pequeno.

**Referencia**:
- [LiveKit: Using a Transformer for Turn Detection](https://blog.livekit.io/using-a-transformer-to-improve-end-of-turn-detection)

---

### Melhoria 8: Mais Dados e Diversidade

**STATUS: PENDENTE**

15K amostras e pouco comparado com o estado da arte:
- Krisp: **2.000 horas / 700K turnos**
- SpeculativeETD: **200+ horas / 122K amostras**
- Pipecat v3.1: **270K amostras / 41GB**

Precisamos de mais diversidade: sotaques (nordestino, gaucho, mineiro), contextos (reuniao formal, papo informal), ruidos de fundo.

**Quem faz isso**:

- **Krisp v2** — melhorou significativamente so mudando os dados, sem mudar arquitetura
- **Pipecat v3.1** — accuracy melhorou dramaticamente ao melhorar o dataset

**Impacto esperado**: +5-10% F1, especialmente em robustez.

**Referencia**:
- [Krisp Turn-Taking v2](https://krisp.ai/blog/krisp-turn-taking-v2-voice-ai-viva-sdk/)
- [Pipecat Smart Turn v3.1 (Daily.co)](https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/)

---

## Plano de Execucao (Ordem de Prioridade)

| # | Melhoria | Status | Impacto | Esforco | Referencia |
|---|----------|--------|---------|---------|------------|
| 1 | Threshold de confianca | FEITO | +11% prec | 0 dias | Pipecat, Krisp |
| 2 | Focal Loss + Label Smoothing | FEITO | Calibracao | 0 dias | Lin et al., EMNLP 2022 |
| 3 | Texto do STT (multimodal) | PENDENTE | +5-10% F1 | 1-2 semanas | LiveKit (-39% FP), Vogent (94.1%) |
| 4 | Hard negatives sinteticos | PENDENTE | +5-15% prec | 1 semana | SpeculativeETD, Deepgram |
| 5 | Labels reais de turn-taking | PENDENTE | +10%+ F1 | 2-4 semanas | SpeculativeETD, Pipecat v3.1 |
| 6 | Contexto conversacional | PENDENTE | +5% F1 | 2-3 semanas | VAP, LiveKit |
| 7 | Knowledge distillation | PENDENTE | +3-7% prec | 1-2 semanas | LiveKit (Qwen 7B→0.5B) |
| 8 | Mais dados / diversidade | PENDENTE | +5-10% F1 | Continuo | Krisp v2, Pipecat v3.1 |

### Meta realista

Combinando melhorias implementadas (1+2) com as proximas (3+4+5): **F1 de 0.80 → 0.92-0.95** com precision acima de 90%.

Os ultimos 5% (0.95 → 1.00) sao os mais dificeis — casos genuinamente ambiguos onde ate humanos discordam. Nenhum projeto no mercado atingiu 100%.

### Proximos passos imediatos

1. **Integrar no BabelCast** com threshold=0.60-0.65 (testar em reunioes reais)
2. **Gerar hard negatives** sinteticos com o TTS do BabelCast (fillers, pausas longas)
3. **Testar modelo do LiveKit** (Qwen2.5-0.5B) como segundo estagio de texto
4. **Baixar dataset do Pipecat v3.1** (270K amostras) e filtrar amostras em portugues

### Infraestrutura

- **Treino**: Modal (A10G GPU, ~$0.50/run de 15-30 min)
- **Deploy**: `modal run modal_finetune.py`
- **Alternativa**: ai-gateway → TensorDock/Vast.ai via `deploy_finetune.py`
- **Modelo final**: 30.5 MB (PyTorch), ~8 MB (ONNX INT8), 12ms inferencia CPU
