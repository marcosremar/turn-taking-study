# Experimento 03 — Fine-tune Pipecat Smart Turn para Portugues + Frances

> Fine-tuning do Pipecat Smart Turn v3 para **avatar conversacional de aprendizado de portugues** por francofonos. Primeiro modelo de turn-taking otimizado para aprendizes L2.

---

## Sumario

1. [Objetivo](#objetivo)
2. [Por que a partir do Pipecat](#por-que-a-partir-do-pipecat)
3. [Metodo — Pipeline de dados](#metodo--pipeline-de-dados)
   - [Pipeline de criacao de dados](#pipeline-de-criacao-de-dados)
   - [Projetos que validam este metodo](#projetos-que-validam-este-metodo)
4. [Fine-tuning](#fine-tuning)
   - [Modelo base](#modelo-base)
   - [Estrategia de fine-tuning](#estrategia-de-fine-tuning)
   - [Infra](#infra)
5. [Melhorias para aprendizado de idiomas (L2)](#melhorias-para-aprendizado-de-idiomas-l2)
   - [Custo assimetrico](#custo-assimetrico)
   - [Threshold duplo (Deepgram Flux)](#threshold-duplo-deepgram-flux)
   - [Dados L2 reais (Speak & Improve)](#dados-l2-reais-speak--improve)
   - [CEFR-aware presets](#cefr-aware-presets)
6. [Sistema de backchannel](#sistema-de-backchannel)
   - [Problema: "o avatar travou?"](#problema-o-avatar-travou)
   - [Solucao: sinais de escuta ativa](#solucao-sinais-de-escuta-ativa)
   - [Presets por nivel CEFR](#presets-por-nivel-cefr)
7. [Engine de inferencia (06_inference.py)](#engine-de-inferencia-06_inferencepy)
   - [Estados do turn-taking](#estados-do-turn-taking)
   - [Exemplo: aprendiz B1 conjugando verbo](#exemplo-aprendiz-b1-conjugando-verbo)
8. [Dados especificos — Frances falando portugues](#dados-especificos--frances-falando-portugues)
   - [Tipos de hesitacao](#tipos-de-hesitacao)
   - [Geracoes com Claude](#geracoes-com-claude)
9. [Benchmarks de referencia](#benchmarks-de-referencia)
   - [Pipecat Smart Turn v3.0](#pipecat-smart-turn-v30)
   - [LiveKit v0.4.1](#livekit-v041)
   - [Outros modelos](#outros-modelos)
   - [Gap sintetico → real](#gap-sintetico--real)
10. [Metricas alvo](#metricas-alvo)
11. [Correcoes apos analise de referencias](#correcoes-apos-analise-de-referencias)
12. [Estrutura de arquivos](#estrutura-de-arquivos)
13. [Cronograma](#cronograma)
14. [Dependencias](#dependencias)
15. [Referencias](#referencias)

---

## Objetivo

Fine-tunar o modelo pre-treinado do **Pipecat Smart Turn v3** (ja treinado em 23 linguas, 270K amostras) especificamente para:

1. **Portugues brasileiro** — melhorar deteccao de fim de turno em conversas em PT-BR
2. **Frances falando portugues** — detectar fim de turno de falantes nativos de frances que estao falando portugues (com sotaque, hesitacoes e code-switching tipicos)
3. **Avatar conversacional L2** — o modelo sera usado em um avatar de IA que ensina portugues a francofonos, exigindo paciencia extra com pausas de aprendiz

> **Pioneirismo**: Nenhum modelo de turn-taking otimizado para aprendizes L2 existe (comercial ou academico). O BabelCast e o primeiro.

## Por que a partir do Pipecat

Nos experimentos anteriores (`previous-experiments/02-finetune-scratch/`) treinamos do zero com Whisper Tiny + dados CORAA/MUPE. Resultados:

| Metrica | Do zero (melhor) | Pipecat original (PT) |
|---------|------------------|----------------------|
| Accuracy | 78.2% | **95.42%** |
| False Positive | 16.8% @0.5 | **2.79%** |
| False Negative | 5.1% @0.5 | **1.79%** |
| F1 | 0.798 | ~0.96 (estimado) |
| Dados | 15K amostras | 270K amostras |
| Labels | Pontuacao + corte artificial | LLM-curados + TTS |

A diferenca e brutal: o Pipecat chega a **95.42% em portugues** com dados LLM-curados + TTS; nos chegamos a 78.2% com labels de pontuacao. O problema nao e o modelo — e a **qualidade dos dados**.

**Problemas do treino do zero:**
- Labels de baixa qualidade (pontuacao nao indica fim de turno de verdade)
- Corte artificial em 30-75% nao simula pausas reais de hesitacao
- Poucos dados (15K vs 270K do Pipecat)
- Modelo nao entende pausas de hesitacao — acerta so 67.7% das pausas @threshold=0.5

**Vantagens de partir do Pipecat:**
- Modelo ja entende turn-taking em 23 linguas
- Precisa de muito menos dados pra adaptar (5-10K vs 270K)
- Transfer learning: mantem conhecimento geral, adapta pra PT-BR
- Treino muito mais rapido (~30 min vs horas)

## Metodo — Pipeline de dados

### Pipeline de criacao de dados

Baseado no pipeline comprovado do Pipecat v3.1:

```
Etapa 1: Frases fonte
  - Transcricoes do CORAA (portugues conversacional real)
  - Frases geradas pelo Claude (contextos especificos)
  - Frases tipicas de aprendiz de frances falando PT
  - Speak & Improve Corpus 2025 (340h L2 com disfluencias)
         |
Etapa 2: LLM processa (Claude Haiku — custo baixo)
  - Filtra frases com erros / ambiguas (Pipecat: Gemini removeu 50-80%)
  - Classifica: COMPLETO vs INCOMPLETO (semantico, nao por pontuacao)
  - Insere fillers brasileiros: "hum", "tipo", "ne", "entao", "e..."
  - Insere fillers de frances falando PT: "euh", "comment dit-on", "como se diz"
  - Gera variantes incompletas: corta frases em pontos naturais
         |
Etapa 3: TTS gera audio
  - Vozes PT-BR nativas (Kokoro, Google Chirp3)
  - Vozes com sotaque frances (XTTS voice cloning, ou Chirp3 com accent)
  - Variacao de velocidade, tom, ruido de fundo
         |
Etapa 4: Dataset final
  - 5-10K amostras balanceadas (50% completo / 50% incompleto)
  - Metadados: lingua, sotaque, tipo_filler, confianca_label
```

### Projetos que validam este metodo

| Projeto | O que fizeram | Resultado |
|---------|--------------|-----------|
| **Pipecat v3.1** | Gemini filtra frases + insere fillers, Claude/GPT geram listas de fillers, Chirp3 TTS | 270K amostras, 81-97% accuracy, 23 linguas |
| **LiveKit** | Qwen 7B como professor gera soft labels, destila pra 0.5B | 99.3% TP rate, -39% interrupcoes |
| **Vogent Turn 80M** | Dados humanos + sinteticos com edge cases (disfluencias, listas, pausas) | 94.1% accuracy, estado da arte |
| **Deepgram** | 100+ horas anotadas por humanos, refinamento iterativo de labels | Melhor calibracao entre todos |
| **SpeculativeETD** | MultiWOZ texto → TTS + fillers injetados + pausas sinteticas | 120K amostras, dataset publico |
| **SODA (Allen AI)** | GPT-3.5 gera 1.5M dialogos a partir de knowledge graph | Preferido sobre BlenderBot, Vicuna |
| **Refuel Autolabel** | GPT-4 como anotador: 88.4% agreement (vs 86% humano) | 20x mais rapido, 7x mais barato |

**Referencia principal:** [Pipecat Data Generation Contribution Guide](https://github.com/pipecat-ai/smart-turn/blob/main/docs/data_generation_contribution_guide.md)

## Fine-tuning

### Modelo base

```python
# Baixar modelo pre-treinado do HuggingFace
from huggingface_hub import hf_hub_download
model_path = hf_hub_download("pipecat-ai/smart-turn-v3", "model.onnx")

# Whisper Tiny encoder (39M) + linear classifier
# Hidden size: 384, ONNX INT8: 8MB, FP32: 32MB
```

### Estrategia de fine-tuning

```python
# Carregar pesos pre-treinados do Pipecat
model = SmartTurnModel(whisper_model="openai/whisper-tiny")
model.load_state_dict(pipecat_weights)

# LR uniforme 5e-5 (igual ao Pipecat train.py)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# Focal Loss com custo assimetrico para aprendizes L2
criterion = FocalLoss(gamma=2.0, alpha=0.25, fp_penalty=2.0)

# 6 epochs, batch_size=128, cosine schedule + warmup 0.2
train(model, portuguese_dataset, epochs=6, batch_size=128)
```

### Infra

- **GPU**: Modal A10G (~$0.50/run)
- **Tempo estimado**: 15-30 min
- **Alternativa**: ai-gateway → TensorDock/Vast.ai

## Melhorias para aprendizado de idiomas (L2)

> Pesquisa realizada em 2026-03-16 confirmou que **nenhum modelo de turn-taking para aprendizes L2 existe** — nem comercial (Praktika, ELSA, Gliglish, TalkPal) nem academico. Todos usam abordagens genericas.

### Custo assimetrico

Para aprendizado de idiomas, **interromper o aluno (FP) e muito pior que esperar demais (FN)**. Um aluno interrompido perde confianca e para de tentar.

```python
# fp_penalty=2.0: falsos positivos custam 2x mais na loss
criterion = FocalLoss(gamma=2.0, alpha=0.25, fp_penalty=2.0)
```

Inspirado no ConversAR (Meta, 2025) que usa "infinite thinking period" para L2.

### Threshold duplo (Deepgram Flux)

Dois thresholds separados para latencia vs precisao:

| Threshold | Valor | Funcao |
|-----------|-------|--------|
| **eager** (0.3-0.5) | Prepara resposta do LLM especulativamente | Reduz latencia percebida |
| **final** (0.7+) | Autoriza o avatar a falar | Minimiza interrupcoes |

Se o score cai antes de atingir `final`, a preparacao especulativa e descartada — sem custo para o aluno.

### Dados L2 reais (Speak & Improve)

O [Speak & Improve Corpus 2025](https://huggingface.co/datasets/speak-improve/corpus) contem **340 horas** de fala L2 em ingles com anotacoes de disfluencia, niveis CEFR A2-C1.

Embora seja ingles (nao portugues), os **padroes de hesitacao L2 sao cross-linguisticos**: pausas longas, repeticoes, auto-correcoes. Usamos como fonte de hesitacao patterns no fine-tuning.

### CEFR-aware presets

O modelo adapta paciencia conforme o nivel do aluno:

| Nivel | final_threshold | eager_threshold | Comportamento |
|-------|----------------|-----------------|---------------|
| **A1** (iniciante) | 0.80 | 0.40 | Muito paciente — espera pausas longas |
| **A2** | 0.75 | 0.38 | Paciente |
| **B1** (intermediario) | 0.70 | 0.35 | Moderado |
| **B2** | 0.65 | 0.33 | Responsivo |
| **C1** (avancado) | 0.60 | 0.30 | Quase nativo — resposta rapida |

## Sistema de backchannel

### Problema: "o avatar travou?"

Aprendizes L2 frequentemente pausam 1-3 segundos para pensar em conjugacoes, vocabulario, ou estrutura. Sem feedback, o aprendiz acha que o avatar travou e desiste de falar.

> **Pesquisa**: Tavus identifica um threshold de **600ms** — apos esse tempo, o falante espera alguma resposta do sistema. Para L2, esse tempo e ainda mais critico.

### Solucao: sinais de escuta ativa

O avatar emite sinais progressivos de que esta ouvindo:

| Tempo de silencio | Sinal | Tipo | Exemplo |
|-------------------|-------|------|---------|
| **600ms** | Aceno visual | Visual | Avatar faz "mhm" com a cabeca |
| **1.5s** | Backchannel verbal | Audio | "mhm", "sim", "uhum" |
| **3.0s** | Encorajamento | Audio | "sem pressa", "pode continuar", "estou ouvindo" |

Os sinais **nao interrompem o turno do aluno** — sao sobreposicoes curtas que indicam escuta ativa.

### Presets por nivel CEFR

| Nivel | Visual (ms) | Verbal (ms) | Encorajamento (ms) |
|-------|-------------|-------------|---------------------|
| **A1** | 500 | 1200 | 2500 |
| **B1** | 600 | 1500 | 3000 |
| **C1** | 800 | 2000 | 4000 |

Alunos A1 recebem sinais mais cedo; alunos C1 precisam de menos apoio.

## Engine de inferencia (06_inference.py)

O arquivo `06_inference.py` implementa a engine completa de turn-taking com backchannel.

### Estados do turn-taking

```
LISTENING → SILENCE → BACKCHANNEL_VISUAL → BACKCHANNEL_VERBAL → ENCOURAGEMENT
    ↑          ↓              ↓                     ↓                  ↓
    ←──────────←──────────────←─────────────────────←──────────────────←
    (aluno volta a falar)

SILENCE → PREPARING → RESPONDING
           (eager)      (final)
```

- **LISTENING**: aluno esta falando, modelo monitora score
- **SILENCE**: pausa detectada, timer inicia
- **BACKCHANNEL_***: sinais de escuta ativa (nao interrompe turno)
- **PREPARING**: score atingiu eager_threshold, LLM comeca a preparar resposta
- **RESPONDING**: score atingiu final_threshold, avatar fala

### Exemplo: aprendiz B1 conjugando verbo

```
Aluno: "Ontem eu... [pausa 600ms]"
Avatar: [aceno visual - nod]
Aluno: "... fui? fiz? [pausa 1.5s]"
Avatar: "mhm" [backchannel verbal]
Aluno: "... fui ao mercado."
Avatar: [espera final_threshold] → responde normalmente
```

## Dados especificos — Frances falando portugues

### Tipos de hesitacao

| Tipo | Exemplo | Label |
|------|---------|-------|
| Filler frances | "Eu fui... euh... ao mercado" | INCOMPLETO |
| Busca de palavra | "Eu preciso de... comment dit-on... uma tesoura" | INCOMPLETO |
| Code-switching | "Eu gosto de... enfin... tipo... de praia" | INCOMPLETO |
| Pausa de conjugacao | "Eu... fui? fiz?... ontem" | INCOMPLETO |
| Entonacao francesa | Frase completa mas com pitch plano (sem queda final) | COMPLETO (dificil) |
| Ritmo silabico | Frances e syllable-timed, PT e stress-timed | Ambos |

### Geracoes com Claude

```
Prompt para Claude gerar frases de aprendiz:

"Gere 100 frases que um frances de nivel B1 falando portugues diria
em uma reuniao de trabalho. Inclua:
- Hesitacoes tipicas (euh, alors, comment dire)
- Erros comuns de conjugacao
- Pausas naturais pra pensar na palavra
- Code-switching involuntario (palavras em frances no meio)
- Frases completas com entonacao plana (sem queda de pitch)

Para cada frase, indique: COMPLETO ou INCOMPLETO"
```

## Benchmarks de referencia

### Pipecat Smart Turn v3.0

Audio-only, 8M params:

| Lingua | Accuracy | FP | FN |
|--------|----------|-----|-----|
| Turco | 97.10% | 1.66% | 1.24% |
| Coreano | 96.85% | 1.12% | 2.02% |
| Japones | 96.76% | 2.04% | 1.20% |
| Frances | 96.01% | 1.60% | 2.39% |
| **Portugues** | **95.42%** | **2.79%** | **1.79%** |
| Ingles | 94.31% | 2.64% | 3.06% |
| Espanhol | 91.97% | 4.48% | 3.55% |

Fonte: [Smart Turn v3 blog](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/)

### LiveKit v0.4.1

Texto-only, 500M params, @99.3% TPR:

| Lingua | TNR | Melhoria vs anterior |
|--------|------|---------------------|
| Hindi | 96.3% | +31.48% |
| Coreano | 94.5% | +30.38% |
| Frances | 88.9% | +33.93% |
| **Portugues** | **87.4%** | **+45.97%** |
| Ingles | 87.0% | +21.69% |

Fonte: [LiveKit v0.4.1 blog](https://livekit.com/blog/improved-end-of-turn-model-cuts-voice-ai-interruptions-39/)

### Outros modelos

| Modelo | Tipo | Accuracy | Linguas | Nota |
|--------|------|----------|---------|------|
| Vogent Turn 80M | Audio+texto | 94.1% | 1 (EN) | Estado da arte multimodal |
| Krisp v2 | Audio | 82.3% bal.acc | "Agnostico" | Proprietario |
| Deepgram Flux | ASR+EoT | #1 VAQI | 1 (EN) | Proprietario |
| VAP | Audio | 79.6% bal.acc | 3 (EN/ZH/JP) | Academico, auto-supervisionado |

### Gap sintetico → real

O SpeculativeETD mostrou que modelos treinados so em dados sinteticos (TTS) perdem **muito** em dados reais:

| Dataset | Wav2vec F1 |
|---------|-----------|
| Sintetico | 94.7% |
| Real | **30.3%** |

Isso mostra que alem de gerar dados com TTS, precisamos incluir **audio real** no fine-tuning.

## Metricas alvo

| Metrica | Exp 02 (do zero) | Pipecat PT (ref) | Alvo (este exp) |
|---------|------------------|------------------|-----------------|
| Accuracy | 78.2% | 95.42% | > 96% |
| False Positive | 16.8% | 2.79% | < 2.5% |
| False Negative | 5.1% | 1.79% | < 2.0% |
| F1 | 0.798 | ~0.96 | > 0.96 |
| Tamanho modelo | 30.5 MB | 8 MB (ONNX INT8) | ~8 MB |
| Inferencia CPU | ~12ms | 12ms | ~12ms |

## Correcoes apos analise de referencias

*(2026-03-16)*

Baixamos 6 papers, 10 blog posts e 7 guias tecnicos (ver `references/`). A analise cruzada revelou problemas no plano original:

| Mudanca | Antes | Depois | Fonte |
|---------|-------|--------|-------|
| Focal Loss alpha | 0.6 | **0.25** | Lin et al. 2017, Table 1a |
| Batch size | 32 | **128** | Pipecat train.py usa 384 |
| Epochs | 10 | **6** | Pipecat usa 4 em 270K |
| Label smoothing | 0.05 | **removido** | EMNLP 2022: dupla regularizacao com FL |
| Learning rate | diferencial (0.1x encoder) | **uniforme 5e-5** | Pipecat train.py usa lr unica |
| Ruido augmentation | Gaussiano | **ruido real** (cafe/escritorio) | Pipecat v3.2: -40% erros |
| ONNX opset | 17 | **18** | Pipecat train.py |
| Quantizacao | nenhuma | **INT8 estatica** (entropy, 1024 calib) | Pipecat deploy: 32MB → 8MB |
| Loss alternativa | so Focal | **Focal + BCE** comparados | Pipecat original usa BCE |

### Tecnicas adicionais identificadas

1. **Knowledge distillation** (LiveKit: -39% interrupcoes, +45.97% melhoria PT)
2. **Short utterance dataset** (Pipecat v3.2: -40% erros em respostas curtas) — implementado
3. **Audio real misturado com TTS** (SpeculativeETD: F1 cai de 94.7% → 30.3% so com sintetico) — implementado via CORAA
4. **Pausa de 1.5-3s apos fillers** (SpeculativeETD V3: melhor variante) — implementado
5. **Threshold per-language** (LiveKit: languages.json com thresholds por lingua)

## Estrutura de arquivos

```
03-finetune-pipecat-pt/
  README.md                    # Este documento
  01_download_pipecat.py       # Baixa modelo pre-treinado do HuggingFace
  02_generate_labels.py        # Claude API gera/filtra/classifica frases PT
  03_generate_audio.py         # TTS gera audio (nativo + sotaque frances)
  04_finetune.py               # Fine-tune com custo assimetrico + dados L2
  05_evaluate.py               # Avaliacao com dual threshold sweep
  06_inference.py              # Engine de inferencia com backchannel + CEFR presets
  modal_run.py                 # Deploy no Modal (GPU)

  references/
    hesitation_turn_taking_l2_review.md    # Mini-artigo com 35 referencias
    papers/                    # 6 PDFs academicos + summaries
      focal_loss_lin_2017.*
      speculative_etd_2025.*
      vap_turn_taking_ekstedt_2024.*
      turn_taking_review_skantze_2021.*
      soda_dialog_distillation_2023.*
      finite_state_turn_taking_raux_2009.*
    papers/hesitation-l2-french/   # 19 PDFs + 10 MD (L2 hesitacao/francofono)
    papers/language-learning-turn-taking/  # Pesquisa L2 turn-taking (2026-03-16)
      survey_turn_taking_iwsds2025.*       # Survey IWSDS 2025
      multilingual_vap_2024.*              # VAP multilingual
      speak_improve_corpus_2025.*          # Speak & Improve L2 corpus
      hesitation_tagging_l2_whisper.*      # Whisper+LoRA hesitation
      conversar_mixed_reality_l2.*         # ConversAR (Meta Quest)
      deepgram_flux.*                      # Deepgram Flux overview
      hume_evi.*                           # Hume EVI overview
      praktika_openai.*                    # Praktika case study
      tavus_turn_taking_guide.*            # Tavus guide
    blogs/                     # 10 blog posts
    guides/                    # 7 technical guides

  data/                        # (gitignored)
    pipecat_pt_audio/          # Audio PT do Pipecat v3.2
    pipecat_pt_test/           # Audio PT test do Pipecat v3.2
    claude_labeled/            # Frases processadas pelo Claude (JSON)
    tts_dataset/               # Audio gerado (nativo + sotaque frances)
    noise_samples/             # Ruido real CC-0 (cafe, escritorio)

  results/                     # (gitignored)
    best_model.pt              # Modelo treinado
    smart_turn_pt_v3.onnx      # ONNX INT8 (~8 MB)
    smart_turn_pt_v3_fp32.onnx # ONNX FP32 (~32 MB)
    training_results.json      # Metricas
    evaluation_results.json    # Comparacao com baseline
```

## Cronograma

| Etapa | Descricao | Tempo |
|-------|-----------|-------|
| 1 | Baixar e inspecionar modelo Pipecat | 1h |
| 2 | Script de labeling com Claude API | 1 dia |
| 3 | Gerar audio TTS (nativo PT-BR) | 1 dia |
| 4 | Gerar audio com sotaque frances | 1-2 dias |
| 5 | Fine-tune no Modal | 30 min |
| 6 | Avaliacao + dual threshold sweep | 2h |
| 7 | Teste no avatar conversacional | 1 dia |
| **Total** | | **~5 dias** |

## Dependencias

```
# Python
torch, torchaudio, transformers    # Modelo
anthropic                          # Claude API (labeling)
kokoro, TTS                        # Geracao de audio
datasets, soundfile, librosa       # Processamento
modal                              # Deploy GPU

# APIs
ANTHROPIC_API_KEY                  # Claude Haiku pra labeling
MODAL_TOKEN                        # Modal pra treino GPU

# Modelos
pipecat-ai/smart-turn-v3          # HuggingFace — modelo pre-treinado
openai/whisper-tiny                # HuggingFace — encoder base
```

## Referencias

Documento completo de pesquisa com 35 referencias: [`references/hesitation_turn_taking_l2_review.md`](references/hesitation_turn_taking_l2_review.md)

**Principais:**

| # | Referencia | Contribuicao |
|---|-----------|--------------|
| 1 | Pipecat Smart Turn v3 (Daily, 2025) | Modelo base, pipeline de dados |
| 2 | Lin et al. (ICCV 2017) | Focal Loss, alpha=0.25 |
| 3 | Skantze (CSL 2021) | Survey de turn-taking |
| 4 | Knill et al. (2025) | Speak & Improve L2 corpus |
| 5 | Saeki et al. (2025) | Whisper+LoRA hesitation tagging |
| 6 | Gamboa et al. (2025) | ConversAR, custo assimetrico L2 |
| 7 | Deepgram Flux (2025) | Dual threshold, speculative ASR |
| 8 | Ekstedt et al. (2024) | VAP multilingual |
| 9 | Raux & Eskenazi (NAACL 2009) | FSM turn-taking |
| 10 | LiveKit (2025) | Knowledge distillation, 500M params |
