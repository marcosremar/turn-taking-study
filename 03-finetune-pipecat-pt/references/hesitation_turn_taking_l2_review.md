# Pausas de Hesitacao, Deteccao de Fim de Turno e Fala L2: Uma Revisao para Fine-Tuning em Portugues

**Contexto**: BabelCast — traducao simultanea em reunioes. O modelo de deteccao de fim de turno precisa distinguir pausas de hesitacao (falante ainda vai continuar) de fim de turno real (pode comecar a traduzir). O desafio e especialmente critico para falantes de frances aprendendo portugues, que produzem pausas longas de hesitacao frequentemente confundidas com fim de turno.

**Autores**: Marcos Remar, com assistencia de Claude (Anthropic)
**Data**: Março 2026

---

## 1. O Problema: Silencio Nao Significa Fim de Turno

Sistemas comerciais de IA conversacional usam thresholds de silencio entre 700ms e 1000ms para detectar fim de turno (Castillo-Lopez, de Chalendar & Semmar, 2025). Esse approach e fundamentalmente inadequado por dois motivos:

1. **Humanos sao mais rapidos**: o gap medio entre turnos em conversacao natural e de apenas ~200ms (Levinson & Torreira, 2015, citado em Skantze, 2021). Esperar 700ms+ resulta em respostas percebidas como lentas.

2. **Pausas dentro de turnos sao frequentemente mais longas que gaps entre turnos** (Skantze, 2021). Um falante pode pausar 2-3 segundos no meio de uma frase (buscando uma palavra ou planejando o proximo trecho) e continuar normalmente. Tratar essa pausa como fim de turno causa interrupcao.

O problema se agrava dramaticamente quando o falante esta usando uma segunda lingua (L2), como demonstrado na literatura a seguir.

---

## 2. Pausas de Hesitacao em Falantes L2

### 2.1 Duracao: L2 pausa 39% mais que nativos

Kosmala & Crible (2022) analisaram pausas preenchidas (filled pauses, FPs) em falantes nativos e nao-nativos de frances, usando o corpus SITAF:

| Metrica | Nativos | Nao-nativos | Diferenca |
|---------|---------|-------------|-----------|
| Duracao media FP | 378ms (SD=200) | **524ms** (SD=222) | +146ms (+39%) |
| Taxa FP/100 palavras | 4.4 | 5.3 | +0.9 (n.s.) |
| Clustering com outros fluencemas | 72% | **82%** | +10 p.p. |

A diferenca de **frequencia** nao e significativa — nao-nativos nao produzem mais pausas, mas pausas **significativamente mais longas** (p < .001). A duracao e um indicador mais confiavel de proficiencia que a frequencia.

Em situacoes de reparo (self-repair), as pausas sao ainda mais extremas. Kosmala (2025) analisou 167 reparos de alunos franceses falando ingles e encontrou:

- Pausa silenciosa media na fase de edicao: **844ms** (SD=573ms)
- Pausa preenchida media na fase de edicao: **522ms** (SD=571ms)
- 82% dos auto-reparos contem uma fase de edicao (pausa e/ou filler)

Esses 844ms estao **muito acima** de qualquer threshold de silencio usado em sistemas comerciais (700ms).

### 2.2 Tipos de pausa: silenciosas dominam

Cenoz (2000) estudou 15 falantes nativos de espanhol falando ingles como L2, analisando 1.085 pausas nao-juntivas:

| Tipo | Proporcao | Faixa de duracao |
|------|-----------|------------------|
| Silenciosas | **64%** | 205ms a 11.569ms |
| Preenchidas | 36% | — |

Distribuicao por duracao das pausas silenciosas:
- 70% entre 200ms e 1.000ms
- 21% entre 1.001ms e 2.000ms
- 7% entre 2.001ms e 4.000ms
- 2% acima de 4.000ms

**Implicacao critica**: a maioria das hesitacoes sao **silencio puro** — nao contem fillers como "euh" ou "hum" que o modelo poderia usar como pista. Isso torna a deteccao mais dificil, pois o modelo precisa distinguir silencio de hesitacao de silencio de fim de turno usando apenas prosodia.

### 2.3 Funcao das pausas: planejamento vs busca lexical

Cenoz (2000) classificou as funcoes das pausas:

| Funcao | Pausas silenciosas | Pausas preenchidas |
|--------|-------------------|-------------------|
| Planejamento (frases, sintaxe) | 59% | **73%** |
| Busca lexical (palavras) | 36% | 26% |
| Outros | 5% | 1% |

Pausas preenchidas sao usadas primariamente para **manter o turno** (floor-holding): 77% ocorrem isoladas, sem outros marcadores de hesitacao. Ja pausas silenciosas co-ocorrem com outras estrategias de reparo 54% das vezes.

### 2.4 Paradoxo de proficiencia

Um achado contraintuitivo: falantes **mais proficientes** produzem **mais** pausas preenchidas, nao menos (Cenoz, 2000). Na faixa de maior proficiencia, a proporcao muda para 53% silenciosas + 46% preenchidas (vs. 69%/31% nos menos proficientes). Isso ocorre porque falantes avancados aprenderam a usar fillers como estrategia de floor-holding — sinalizam ao interlocutor que ainda estao falando.

A variacao individual e enorme: a proporcao de pausas preenchidas varia de **4% a 74.5%** entre falantes individuais.

---

## 3. Fillers Franceses na Fala L2

### 3.1 Transferencia linguistica

Falantes franceses transferem seus fillers nativos para a L2. Lo (2018) analisou 15 bilingues alemao-frances e mostrou que as propriedades acusticas dos fillers sao distintas por lingua:

- **Frances "euh"**: duracao mais longa, F1-F3 mais altos (vogal central schwa)
- **Alemao "ah"**: duracao mais curta, F1-F3 mais baixos (vogal aberta)

Bilingues mantem formas foneticamente distintas para cada lingua, indicando que **a forma do filler revela qual lingua esta sendo processada**. Isso e confirmado por Kosmala (2025): alunos franceses falando ingles usam "euh" (24 instancias) e "eum" (12 instancias), os fillers franceses, nao os ingleses.

### 3.2 Fillers sao itens lexicais, nao sons universais

Bottcher & Zellers (2024) analisaram o corpus RUEG (736 falantes, 5 linguas, 4.468 narracoes) e demonstraram que fillers sao **itens lexicais especificos de cada lingua**, nao sons universais de hesitacao. A proporcao nasal ("uhm") vs. nao-nasal ("uh") varia por lingua, genero e idade.

Falantes heritage (bilingues desde a infancia) produzem **mais fillers no total** devido a carga cognitiva da ativacao de duas linguas. A **tolerancia ao silencio** tambem se transfere da L1: falantes de japones mantem maior tolerancia ao silencio mesmo falando ingles.

### 3.3 Code-switching: 93% nas fronteiras de unidade

Beatty-Martinez, Navarro-Torres & Dussias (2020) analisaram 10 bilingues espanhol-ingles e encontraram que **93% das trocas de codigo ocorrem em fronteiras de unidades entonacionais** — os mesmos pontos onde transicoes de turno acontecem. Antes de uma troca, ha **reorganizacao prosodica** (mudanca na velocidade de fala), que pode ser confundida com marcadores prosodicos de fim de turno.

**Implicacao para o modelo**: quando um frances falando portugues alterna para uma palavra em frances ("Eu preciso de... *comment dit-on*... uma tesoura"), a mudanca prosodica antes da troca NAO indica fim de turno.

---

## 4. Pistas Prosodicas de Fim de Turno

### 4.1 O contorno de F0 e o sinal-chave

Ekstedt & Skantze (2022) conduziram experimentos de perturbacao prosodica com o modelo VAP para isolar a importancia relativa de cada pista:

| Pista prosodica | Importancia |
|----------------|-------------|
| Informacao fonetica/espectral | Mais importante no geral |
| **Contorno de F0 (pitch)** | **Mais importante para desambiguar pontos sintaticamente completos** |
| Intensidade | Comparavel ao F0 no geral |
| Normalizacao de duracao | Menos importante individualmente |

O achado central: **F0 e o principal desambiguador em pontos onde a sintaxe poderia indicar completude**. Um F0 ascendente + silaba final mais longa = sinal de completude de turno. O modelo VAP consegue prever trocas de turno **antes** da completude do enunciado a partir da dinamica prosodica.

### 4.2 Prosodia antes de pausas preenchidas

Wu, Didirkova & Simon (2023) analisaram o corpus LOCAS-F (2h38m, >1.000 FPs, kappa inter-anotador = 0.86) e encontraram padroes prosodicos especificos antes de fillers:

- Queda de F0 de **1-2 semitons** durante pausas preenchidas (17.90 vs. 19.01 ST)
- Reset melodico **negativo significativo** antes da FP (descontinuidade de pitch)
- Efeito de preparacao: queda de **4.66 ST** entre fala nao-preparada (18.89 ST) e preparada (14.24 ST)
- **Nao ha reset significativo** entre a FP e a silaba seguinte

**Implicacao**: a queda de pitch antes de um filler "euh" e diferente da queda de pitch de fim de frase. O modelo precisa aprender essa distincao sutil — o que o encoder do Whisper pode capturar atraves de representacoes espectrais.

### 4.3 Resumo das pistas prosodicas de turno

Skantze (2021) compilou a literatura:

| Pista | Fim de turno | Pausa de hesitacao |
|-------|-------------|-------------------|
| Pitch (F0) | Queda (declarativa) ou queda-ascensao | Suspenso ou queda menor |
| Intensidade | Diminui gradualmente | Mantem-se ou cai abruptamente |
| Velocidade de fala | Desacelera antes do fim | Pode acelerar antes de parar |
| Alongamento silabico | Silaba final alongada | Ausente |
| Completude sintatica | Frase sintaticamente completa | Frase incompleta |

---

## 5. Estado da Arte em Deteccao de Fim de Turno

### 5.1 Modelos e suas acuracias

| Modelo | Tipo | Params | Tamanho | Latencia | Acuracia | Linguas |
|--------|------|--------|---------|----------|----------|---------|
| **Pipecat Smart Turn v3.1** | Audio (Whisper Tiny) | 8M | 8 MB (INT8) | 12ms CPU | 94.7% (EN) | 23 |
| **LiveKit v0.4.1** | Texto (Qwen2.5-0.5B) | 500M | — | — | TNR 87.4% (PT) | 15+ |
| **Vogent-Turn-80M** | Multimodal | 79.2M | — | 7ms T4 | 94.1% (EN) | 1 |
| **Krisp TT v1** | Audio | 6.1M | 65 MB | — | 82% bal.acc. | 1 |
| **VAP** | Audio (CPC+Transformer) | — | — | 14.6ms CPU | 76.2% bal.acc. | 3 |
| **SpeculativeETD** | Audio (GRU+Wav2vec) | 1M+94M | — | 0.26ms+server | 66% real | 1 |

Nenhum destes modelos foi especificamente otimizado para portugues ou para fala L2.

### 5.2 Pipecat Smart Turn: resultados por lingua

| Lingua | Acuracia | FP | FN |
|--------|----------|-----|-----|
| Turco | 97.10% | 1.66% | 1.24% |
| Coreano | 96.85% | 1.12% | 2.02% |
| Japones | 96.76% | 2.04% | 1.20% |
| Frances | 96.01% | 1.60% | 2.39% |
| **Portugues** | **95.42%** | **2.79%** | **1.79%** |
| Ingles | 94.31% | 2.64% | 3.06% |
| Espanhol | 91.97% | 4.48% | 3.55% |

Portugues esta atras de frances, japones, coreano e turco. A taxa de falsos positivos (2.79%) indica que o modelo diz "terminou" quando nao terminou em quase 3% dos casos — problematico para traducao simultanea.

### 5.3 O gap sintetico → real: o maior risco

Ok, Yoo & Lee (2025) demonstraram um colapso devastador quando modelos treinados em dados sinteticos (TTS) sao avaliados em fala real:

| Modelo | F1 sintetico | F1 real | Perda |
|--------|-------------|---------|-------|
| Wav2vec 2.0 | 94.7% | **30.3%** | -64.4 p.p. |
| SpeculativeETD | 88.9 IoU | **16.4 IoU** | -72.5 p.p. |
| VAP | 87.7 IoU | **10.7 IoU** | -77.0 p.p. |

**Todos os modelos colapsam** ao sair de dados sinteticos para conversacao real. A melhoria de v3.0 para v3.1 do Pipecat veio justamente de adicionar **audio humano real** de tres parceiros especializados (Liva AI, Midcentury, MundoAI).

---

## 6. Tecnicas de Treinamento Validadas

### 6.1 Focal Loss (Lin et al., 2017)

A Focal Loss resolve o desbalanceamento extremo de classes (a maioria dos frames e "fala em andamento"; fim de turno e raro):

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

Com gamma=2, a perda para exemplos bem classificados (p_t=0.9) e **100x menor** que cross-entropy padrao; com p_t=0.968, **1.000x menor**. Os melhores hiperparametros: **gamma=2, alpha=0.25** (Lin et al., 2017, Tabela 1a).

### 6.2 Knowledge Distillation (LiveKit)

O LiveKit treinou um modelo professor (Qwen2.5-7B) e destilou para um aluno (Qwen2.5-0.5B), que converge apos ~1.500 steps. O resultado: reducao de 39% nos falsos positivos de interrupcao. Para portugues especificamente, a melhoria relativa foi de **45.97%** — a segunda maior entre todas as linguas.

### 6.3 Dados curtos e ruidosos (Pipecat v3.2)

O Pipecat v3.2 adicionou dois tipos de dados ao treinamento:
1. **Respostas curtas** ("yes", "no", "ok"): reduziu erros de classificacao em **40%**
2. **Ruido de fundo** (cafe, escritorio, CC-0 Freesound): melhorou robustez

### 6.4 Injecao de fillers e pausas (SpeculativeETD)

Ok et al. (2025) testaram tres variantes de dados sinteticos:
- V1: TTS direto (baseline)
- V2: Pausas de hesitacao estendidas para 1.5-3.0 segundos
- **V3: Fillers injetados ("um", "uh") em posicoes aleatorias + pausas apos**

V3 foi a melhor variante, validando a abordagem de gerar dados com fillers inseridos por LLM.

### 6.5 Transfer learning cross-lingual

Castillo-Lopez et al. (2025) documentam que VAP pre-treinado em ingles (Switchboard) e fine-tunado em japones **supera** o treinamento direto em japones. Isso valida a abordagem de partir do Pipecat (pre-treinado em 23 linguas) e fine-tunar para portugues.

---

## 7. Pesquisa: Modelos de Turn-Taking para Aprendizado de Idiomas (Marco 2026)

**Data da pesquisa**: 16 de marco de 2026
**Objetivo**: identificar modelos e tecnicas especificas para turn-taking em contexto de aprendizado de idiomas com avatar conversacional

### 7.1 Panorama: nenhum modelo especifico existe

Nao existe nenhum modelo open-source de turn-taking treinado especificamente para aprendizes L2. Os produtos comerciais usam abordagens genericas:

| Produto | Abordagem | Limitacao |
|---------|-----------|-----------|
| **Praktika** (OpenAI) | GPT-5.2 Realtime API — turn detection nativo OpenAI | Caixa preta, sem adaptacao L2 |
| **ConversAR** (Meta Quest) | Timeout fixo 2000ms + "periodo infinito de pensamento" | Nao distingue hesitacao de fim de turno |
| **Gliglish** | ChatGPT + speech recognition generico | Sem turn detection especializado |
| **ELSA Speak** | Modelo proprietario treinado com sotaques variados | Foco em pronuncia, nao turn-taking |
| **TalkPal/SpeakPal/Talkio** | GPT + NLP generico | Sem modelo de audio para end-of-turn |
| **Hume EVI** | Prosodia (tom de voz) para turn detection | Comercial, sem foco em L2 |
| **Deepgram Flux** | Modelo fusionado (ASR + turn detection) com `eot_threshold` configuravel | So ingles, sem adaptacao para proficiencia |

### 7.2 Modelos de pesquisa relevantes

- **VAP (Voice Activity Projection)** — multilingual (EN/ZH/JA), fine-tuning por lingua melhora significativamente, mas nao cobre L2 ou non-native speakers (Inoue et al., 2024)
- **Speak & Improve Corpus 2025** — 340 horas de fala L2 ingles com anotacao de disfluencias e CEFR scores (A2-C1, maioria B1-B2) (Knill et al., 2025)
- **Whisper + LoRA para hesitation tagging** — fine-tune do Whisper Large V3 com tags de hesitacao acusticamente precisas: **11.3% melhoria no WER** para fala L2 (arXiv:2506.04076)
- **Deepgram Flux** — modelo fusionado (ASR + turn detection em um unico modelo); ~260ms end-of-turn detection, ~30% menos interrupcoes vs pipeline tradicional (Deepgram, 2025)
- **Survey IWSDS 2025** — 72% dos trabalhos de turn-taking NAO comparam com metodos anteriores, sugerindo falta de benchmarks estabelecidos (Castillo-Lopez et al., 2025)

### 7.3 Tecnicas identificadas para melhorar o fine-tuning

**1. Threshold duplo (Eager + Final) — inspirado no Deepgram Flux**

Em vez de um unico threshold, usar dois:
- **Eager threshold (0.3-0.5)**: o avatar comeca a preparar resposta especulativamente (inicia geracao LLM)
- **Final threshold (0.7+)**: confirma fim de turno e fala

Economia de latencia estimada: 150-250ms no inicio da resposta, sem interromper o falante.

Deepgram Flux implementa isso como `eot_threshold` (padrao 0.7) e `eager_eot_threshold` (0.3-0.5), com parametro `eot_timeout_ms` para configurar sensibilidade.

**2. Custo assimetrico (FP >> FN)**

Para um tutor de idiomas, **interromper o aluno e muito pior que esperar demais**:
- ConversAR (2025): "periodo infinito de pensamento" — learners controlam quando falar
- Praktika: timeout estendido para fala L2 fragmentada e com sotaque
- Implementacao: `fp_penalty=2.0` no focal loss — erros de interrupcao custam 2x mais

**3. Dados L2 reais — Speak & Improve Corpus**

O corpus Speak & Improve 2025 (Knill et al., 2025) tem:
- 340 horas de audio L2 ingles
- Anotacao de disfluencias (false starts, repeticoes, hesitacoes)
- Scores CEFR: B1 (18.3%), B1+ (25.3%), B2 (25.1%), B2+ (18.3%)
- Embora seja ingles L2, padroes de hesitacao L2 sao transferiveis entre linguas (Cenoz, 2000)

Fine-tuning do Whisper com anotacao precisa de hesitacoes (tags "um"/"uh" acusticamente alinhadas) mostrou 11.3% melhoria no WER vs ignorar hesitacoes.

**4. Proficiency-aware turn-taking (futuro)**

Nenhum sistema implementa isso ainda, mas e logico:
- Aluno A1: pausa **3-5x mais** que B2
- Aluno B2: usa fillers como estrategia de floor-holding (Cenoz, 2000)
- De Jong & Bosker: threshold otimo de 250ms para pausa L2 em holandes
- Shea & Leonard (2019): threshold de **1000ms** para espanhol L2
- Implementacao futura: input de nivel CEFR ao modelo, ou adaptar threshold baseado em perfil do falante

### 7.5 Sistema de backchannel para avatar de aprendizado

Um problema critico identificado: se o avatar espera 3 segundos em silencio, o aprendiz pensa que o sistema travou e para de falar. A solucao e um sistema de **backchannel signals** que mostra ao aprendiz que o avatar esta ouvindo, sem tomar o turno.

| Tempo de silencio | Acao do avatar | Tipo |
|-------------------|----------------|------|
| 0-600ms | Nada (normal) | — |
| 600ms-1.5s | Aceno de cabeca, olhar atento | Backchannel visual |
| 1.5s-3.0s | "Mhm...", "Continue...", "Uhum..." | Backchannel verbal |
| 3.0s+ | "Sem pressa, pode pensar...", "Prenez votre temps..." | Encorajamento |

O sistema integra o threshold duplo do modelo com os backchannels:
- **Eager threshold** atingido → avatar da backchannel + inicia geracao LLM especulativa
- **Final threshold** atingido → avatar fala a resposta

Presets por nivel CEFR:

| CEFR | Eager | Final | BC Visual | BC Verbal | Encorajamento |
|------|-------|-------|-----------|-----------|---------------|
| A1 | 0.50 | **0.80** | 500ms | 1200ms | 2500ms |
| A2 | 0.45 | 0.75 | 550ms | 1300ms | 2800ms |
| B1 | 0.40 | 0.70 | 600ms | 1500ms | 3000ms |
| B2 | 0.35 | 0.65 | 600ms | 1800ms | 4000ms |
| C1 | 0.35 | **0.60** | 600ms | 2000ms | 5000ms |

O avatar e mais paciente com A1/A2 (threshold final 0.80 = raramente interrompe) e mais responsivo com B2/C1 (0.60-0.65 = conversa mais natural). Implementado em `06_inference.py`.

### 7.6 Conclusao da pesquisa

O trabalho do BabelCast e confirmado como **pioneiro**: nao existe modelo de turn-taking para aprendizes L2, muito menos para francofonos falando portugues. As melhorias implementadas (custo assimetrico, threshold duplo, dados L2 reais, backchannel por CEFR) trazem o modelo mais proximo do comportamento ideal de um tutor paciente.

---

## 8. Implicacoes para o Fine-Tuning BabelCast

### 8.1 O que o modelo precisa aprender

Com base na literatura, os seguintes padroes sao criticos para o modelo distinguir:

| Padrao | Duracao tipica | Label | Pista prosodica |
|--------|---------------|-------|----------------|
| Fim de turno real | silencio 200-500ms | COMPLETO | F0 caindo, intensidade caindo, silaba final alongada |
| Hesitacao nativa (PT-BR) | FP 378ms + silencio 200-1000ms | INCOMPLETO | F0 suspenso, "hum"/"tipo"/"ne" |
| Hesitacao L2 (francofono) | FP **524ms** + silencio **844ms** | INCOMPLETO | F0 suspenso, "euh"/"alors"/"comment dire" |
| Busca de palavra L2 | silencio 1000-3000ms | INCOMPLETO | Silencio puro, sem pista prosodica clara |
| Code-switching | prosodic reset + silencio 500-1500ms | INCOMPLETO | Mudanca de velocidade antes da troca |
| Resposta curta completa | "Sim." + silencio 200ms | COMPLETO | F0 caindo, curta duracao |
| Resposta curta incompleta | "Sim, mas..." + silencio | INCOMPLETO | F0 suspenso ou ascendente |

### 8.2 Dados necessarios

1. **Audio real de portugues** (CORAA MUPE 365h, NURC-SP 239h) — evitar o gap sintetico→real
2. **TTS com fillers** brasileiros ("hum", "tipo", "ne") e franceses ("euh", "alors") inseridos por LLM
3. **Pausas longas de hesitacao** (1.5-3.0s) injetadas em amostras incompletas
4. **Respostas curtas** ("sim", "nao", "ok", "tá bom") com variantes completas e incompletas
5. **Code-switching** frances-portugues em fronteiras de unidade
6. **Ruido de fundo** real (cafe, escritorio) — CC-0 Freesound

### 8.3 Escolhas arquiteturais validadas

- **Whisper Tiny encoder** (39M params, 8s janela, 384-dim): mesma arquitetura do Pipecat, captura prosodia sem decodificar texto
- **Attention pooling**: aprende quais frames perto do silencio sao mais informativos (Ekstedt & Skantze, 2022)
- **Focal Loss** (gamma=2, alpha=0.25): calibracao implícita sem label smoothing (EMNLP 2022)
- **INT8 quantizacao**: 32MB → 8MB, 12ms CPU (Pipecat deploy pipeline)

### 8.4 Lacuna na literatura

**Nenhum paper foi encontrado especificamente sobre deteccao de fim de turno em fala L2**, e especialmente nao sobre francofonos falando portugues. A intersecao de:
- Deteccao de fim de turno (ML/audio)
- Pausas de hesitacao em L2 (linguistica)
- Fala francofona em portugues (fonologia)

...e uma area completamente inexplorada. O trabalho do BabelCast e, ate onde sabemos, o primeiro a atacar esse problema especifico.

---

## Referencias

### Papers Academicos

1. Beatty-Martinez, A. L., Navarro-Torres, C. A., & Dussias, P. E. (2020). Codeswitching: A bilingual toolkit for opportunistic speech planning. *Frontiers in Psychology*, 11, 1699.

2. Bottcher, A. & Zellers, M. (2024). Do you say uh or uhm? A cross-linguistic approach to filler particle use in heritage and majority speakers across three languages. *Frontiers in Psychology*, 15, 1358182.

3. Castillo-Lopez, V., de Chalendar, G., & Semmar, N. (2025). A survey of recent advances on turn-taking modeling in spoken dialogue systems. *arXiv:2503.xxxxx*.

4. Cenoz, J. (2000). Pauses and hesitation phenomena in second language production. *ITL - International Journal of Applied Linguistics*, 127-128, 53-69.

5. Christodoulides, G. & Avanzi, M. (2014). DisMo: A morphosyntactic, disfluency and multi-word unit annotator. *Proceedings of LREC 2014*.

6. Christodoulides, G. & Avanzi, M. (2015). Automatic detection and annotation of disfluencies in spoken French corpora. *Proceedings of Interspeech 2015*.

7. Ekstedt, E. & Skantze, G. (2022). How much does prosody help turn-taking? Investigations using voice activity projection models. *Proceedings of Interspeech 2022*.

8. Inoue, K., Jiang, D., Ekstedt, E., Kawahara, T., & Skantze, G. (2024). Real-time and continuous turn-taking prediction using voice activity projection. *arXiv:2401.04868*.

9. Inoue, K., Jiang, D., Ekstedt, E., Kawahara, T., & Skantze, G. (2024). Multilingual turn-taking prediction using voice activity projection. *Proceedings of LREC-COLING 2024*.

10. Kosmala, L. & Crible, L. (2022). The dual status of filled pauses: Evidence from genre, proficiency and co-occurrence. *Language and Speech*, 65(4), 1-25.

11. Kosmala, L. (2023). Exploring the status of filled pauses as pragmatic markers: The role of gaze and gesture. *Journal of Pragmatics*, 212, 1-15.

12. Kosmala, L. (2025). Multimodal self- and other-initiated repairs in L2 peer interactions. *Proceedings of DiSS 2025*, Lisbon.

13. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal loss for dense object detection. *Proceedings of ICCV 2017*. arXiv:1708.02002.

14. Lo, S. L. (2018). Between ah(m) and euh(m): Filled pauses in German-French bilinguals. *BAAP 2018 Poster Presentation*.

15. Ok, J., Yoo, I. C., & Lee, Y. (2025). Speculative end-turn detector: Revisiting an efficient design for low-latency end-of-turn detection. *arXiv:2503.23439*.

16. Peters, M. (2017). L2 fluency development in French. *Ph.D. Thesis*.

17. Raux, A. & Eskenazi, M. (2009). A finite-state turn-taking model for spoken dialog systems. *Proceedings of NAACL-HLT 2009*.

18. Skantze, G. (2021). Turn-taking in conversational systems and human-robot interaction: A review. *Computer Speech & Language*, 67, 101178.

19. Wu, X., Didirkova, I., & Simon, A. C. (2023). Disfluencies in continuous speech in French: Prosodic parameters of filled pauses and vowel lengthening. *Proceedings of ICPhS 2023*.

20. Knill, K., et al. (2025). Speak & Improve Corpus 2025: an L2 English Speech Corpus for Language Assessment and Feedback. *arXiv:2412.11986*.

21. Saeki, T., et al. (2025). Acoustically precise hesitation tagging is essential for end-to-end verbatim transcription systems. *arXiv:2506.04076*.

22. Gamboa, H. & Wohlgenannt, G. (2025). ConversAR: Practicing a second language without fear — Mixed reality agents for interactive group conversation. *arXiv:2510.08227*.

23. De Jong, N. & Bosker, H. R. (2013). Choosing a threshold for silent pauses to measure second language fluency. *The 6th Workshop on Disfluency in Spontaneous Speech (DiSS)*.

24. Shea, C. & Leonard, K. (2019). Evaluating measures of pausing for second language fluency research. *Journal of Second Language Pronunciation*, 5(2), 254-277.

### Modelos e Blogs Tecnicos

20. Daily/Pipecat. (2025). Announcing Smart Turn v3, with CPU inference in just 12ms. https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/

21. Daily/Pipecat. (2025). Improved accuracy in Smart Turn v3.1. https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/

22. Daily/Pipecat. (2026). Smart Turn v3.2: Handling noisy environments and short responses. https://www.daily.co/blog/smart-turn-v3-2-handling-noisy-environments-and-short-responses/

23. LiveKit. (2025). Improved end-of-turn model cuts voice AI interruptions 39%. https://livekit.com/blog/improved-end-of-turn-model-cuts-voice-ai-interruptions-39/

24. LiveKit. (2025). Using a transformer to improve end-of-turn detection. https://blog.livekit.io/using-a-transformer-to-improve-end-of-turn-detection

25. Krisp. (2025). Audio-only 6M weights turn-taking model for voice AI agents. https://krisp.ai/blog/turn-taking-for-voice-ai/

26. Deepgram. (2025). Evaluating end-of-turn detection models. https://deepgram.com/learn/evaluating-end-of-turn-detection-models

27. Vogent. (2025). Vogent-Turn-80M model card. https://huggingface.co/vogent/Vogent-Turn-80M

28. Deepgram. (2025). Introducing Flux: Conversational Speech Recognition. https://deepgram.com/learn/introducing-flux-conversational-speech-recognition

29. Hume AI. (2025). Empathic Voice Interface (EVI) documentation. https://dev.hume.ai/docs/speech-to-speech-evi/overview

30. OpenAI. (2025). Inside Praktika's conversational approach to language learning. https://openai.com/index/praktika/

31. Tavus. (2025). The Complete Guide to AI Turn-Taking. https://www.tavus.io/post/ai-turn-taking

### Datasets

32. Pipecat-AI. (2026). Smart Turn Data v3.2 Training Set. HuggingFace: `pipecat-ai/smart-turn-data-v3.2-train`. 270,946 samples, 23 languages.

33. NILC-NLP. CORAA-MUPE-ASR. HuggingFace: `nilc-nlp/CORAA-MUPE-ASR`. 365h, Portuguese interviews.

34. NILC-NLP. CORAA NURC-SP Audio Corpus. HuggingFace: `nilc-nlp/CORAA-NURC-SP-Audio-Corpus`. 239h, Portuguese dialogues.

35. Cambridge English. Speak & Improve Corpus 2025. 340h, L2 English learner speech with disfluency annotations and CEFR scores.
