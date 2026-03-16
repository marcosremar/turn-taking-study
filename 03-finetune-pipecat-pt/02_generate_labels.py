"""Generate high-quality turn-taking labels using Claude API.

Based on the Pipecat v3.1 pipeline:
1. Load Portuguese transcripts from CORAA MuPe + NURC-SP
2. Claude filters bad/ambiguous sentences (Pipecat: Gemini removed 50-80%)
3. Claude classifies: COMPLETE vs INCOMPLETE (semantic, not punctuation-based)
4. Claude inserts Brazilian Portuguese fillers
5. Claude generates French-accented Portuguese variants

Uses Claude Haiku for cost efficiency (~$0.001 per 20 sentences).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = Path("/workspace/hf_cache") if Path("/workspace").exists() else DATA_DIR / "hf_cache"

# Brazilian Portuguese fillers (from research: Claude + GPT generate these for Pipecat)
PT_BR_FILLERS = [
    "hum", "eh", "ah", "tipo", "ne", "entao", "assim", "quer dizer",
    "como e que eu falo", "deixa eu pensar", "olha", "bom", "pois e",
    "sabe", "veja bem", "na verdade", "digamos", "e...",
]

# French speaker fillers when speaking Portuguese
FR_PT_FILLERS = [
    "euh", "alors", "comment dire", "como se diz", "enfin",
    "c'est-a-dire", "voila", "bon", "donc",
]


@dataclass
class LabeledSentence:
    text: str
    label: str  # "complete" or "incomplete"
    confidence: float  # 0-1
    source: str  # "coraa", "mupe", "claude_generated"
    has_filler: bool = False
    filler_type: str = ""  # "pt_br" or "fr_pt"
    original_text: str = ""


def load_coraa_transcripts(max_samples: int = 10000) -> list[dict]:
    """Load transcripts from CORAA MuPe (365h interviews, diarization kappa 0.947)."""
    from datasets import load_dataset

    log.info("Loading CORAA MuPe transcripts (streaming)...")
    ds = load_dataset(
        "nilc-nlp/CORAA-MUPE-ASR",
        split="train",
        streaming=True,
        cache_dir=str(CACHE_DIR),
    )

    transcripts = []
    for i, row in enumerate(ds):
        text = str(row.get("text", row.get("normalized_text", "")))
        if not text or len(text.split()) < 3:
            continue

        transcripts.append({
            "text": text,
            "speaker_type": row.get("speaker_type", ""),
            "speaker_code": str(row.get("speaker_code", f"mupe_{i}")),
            "source": "mupe",
        })

        if len(transcripts) >= max_samples:
            break

        if i % 5000 == 0 and i > 0:
            log.info("  Scanned %d rows, collected %d transcripts", i, len(transcripts))

    log.info("CORAA MuPe: %d transcripts loaded", len(transcripts))
    return transcripts


def load_nurc_transcripts(max_samples: int = 10000) -> list[dict]:
    """Load transcripts from CORAA NURC-SP (239h dialogues, speaker IDs)."""
    from datasets import load_dataset

    log.info("Loading CORAA NURC-SP transcripts (streaming)...")
    try:
        ds = load_dataset(
            "nilc-nlp/CORAA-NURC-SP-Audio-Corpus",
            split="train",
            streaming=True,
            cache_dir=str(CACHE_DIR),
        )
    except Exception as e:
        log.warning("Failed to load NURC-SP: %s", e)
        return []

    transcripts = []
    for i, row in enumerate(ds):
        text = str(row.get("text", row.get("sentence", "")))
        if not text or len(text.split()) < 3:
            continue

        transcripts.append({
            "text": text,
            "speaker_type": row.get("speech_genre", ""),
            "speaker_code": str(row.get("speaker_id", f"nurc_{i}")),
            "source": "nurc_sp",
        })

        if len(transcripts) >= max_samples:
            break

    log.info("NURC-SP: %d transcripts loaded", len(transcripts))
    return transcripts


def classify_with_claude(
    sentences: list[str],
    batch_size: int = 20,
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict]:
    """Use Claude to classify sentences as complete/incomplete.

    Based on Pipecat's approach: Gemini 2.5 Flash filtered 50-80% of sentences.
    We use Claude Haiku for cost efficiency.

    Returns list of {text, label, confidence, keep}.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set — using rule-based fallback")
        return _rule_based_classify(sentences)

    client = anthropic.Anthropic(api_key=api_key)
    results = []

    for batch_start in range(0, len(sentences), batch_size):
        batch = sentences[batch_start:batch_start + batch_size]
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch))

        prompt = f"""Voce e um anotador de turn-taking para portugues brasileiro.
Para cada frase abaixo, classifique:
- COMPLETO: o falante terminou o pensamento (pode comecar a traduzir)
- INCOMPLETO: o falante vai continuar falando (nao traduzir ainda)
- RUIM: frase com erro, ambigua, ou inutilizavel (descartar)

Responda APENAS em JSON, sem explicacao:
[{{"n": 1, "label": "COMPLETO", "confidence": 0.95}}, ...]

Frases:
{numbered}"""

        try:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text

            # Parse JSON from response
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                batch_results = json.loads(json_match.group())
                for item in batch_results:
                    idx = item["n"] - 1
                    if 0 <= idx < len(batch):
                        results.append({
                            "text": batch[idx],
                            "label": item["label"].lower(),
                            "confidence": item.get("confidence", 0.5),
                            "keep": item["label"].upper() != "RUIM",
                        })
            else:
                log.warning("No JSON in Claude response for batch %d", batch_start)
                results.extend(_rule_based_classify(batch))

        except Exception as e:
            log.warning("Claude API error at batch %d: %s — falling back to rules", batch_start, e)
            results.extend(_rule_based_classify(batch))

        # Rate limiting
        if batch_start % 100 == 0 and batch_start > 0:
            log.info("  Classified %d/%d sentences", batch_start, len(sentences))
            time.sleep(0.5)

    return results


def _rule_based_classify(sentences: list[str]) -> list[dict]:
    """Fallback rule-based classification (less accurate than Claude)."""
    results = []
    for s in sentences:
        text = s.strip()
        if not text or len(text) < 5:
            results.append({"text": text, "label": "ruim", "confidence": 0.5, "keep": False})
            continue

        if re.search(r'[.!?]+\s*$', text):
            results.append({"text": text, "label": "completo", "confidence": 0.7, "keep": True})
        elif re.search(r'[,;:\-]\s*$', text):
            results.append({"text": text, "label": "incompleto", "confidence": 0.6, "keep": True})
        elif re.search(r'\b(e|que|mas|porque|quando|se|como|pra|para)\s*$', text, re.I):
            results.append({"text": text, "label": "incompleto", "confidence": 0.8, "keep": True})
        else:
            results.append({"text": text, "label": "completo", "confidence": 0.5, "keep": True})

    return results


def insert_fillers_with_claude(
    sentences: list[str],
    filler_type: str = "pt_br",
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict]:
    """Use Claude to insert fillers at natural points.

    Based on Pipecat: Gemini Flash inserted fillers at natural break points.

    filler_type: "pt_br" for native Brazilian, "fr_pt" for French-accented
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set — using simple filler insertion")
        return _simple_filler_insert(sentences, filler_type)

    client = anthropic.Anthropic(api_key=api_key)
    fillers = PT_BR_FILLERS if filler_type == "pt_br" else FR_PT_FILLERS

    filler_list = ", ".join(f'"{f}"' for f in fillers)

    if filler_type == "fr_pt":
        context = """O falante e um frances de nivel B1 falando portugues.
Ele hesita ao conjugar verbos, busca palavras, e as vezes usa fillers em frances.
Insira hesitacoes naturais como: """ + filler_list
    else:
        context = """O falante e brasileiro nativo.
Insira fillers naturais do portugues brasileiro como: """ + filler_list

    results = []
    batch_size = 15

    for batch_start in range(0, len(sentences), batch_size):
        batch = sentences[batch_start:batch_start + batch_size]
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch))

        prompt = f"""{context}

Para cada frase, crie uma versao com 1-2 fillers inseridos em pontos naturais.
A frase deve parecer INCOMPLETA (como se o falante fosse continuar).

Responda APENAS em JSON:
[{{"n": 1, "original": "frase original", "with_filler": "frase com filler"}}]

Frases:
{numbered}"""

        try:
            response = client.messages.create(
                model=model,
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                batch_results = json.loads(json_match.group())
                for item in batch_results:
                    results.append({
                        "original": item.get("original", ""),
                        "with_filler": item.get("with_filler", ""),
                        "filler_type": filler_type,
                    })

        except Exception as e:
            log.warning("Claude filler API error: %s", e)
            results.extend(_simple_filler_insert(batch, filler_type))

        time.sleep(0.3)

    return results


def _simple_filler_insert(sentences: list[str], filler_type: str) -> list[dict]:
    """Simple rule-based filler insertion fallback."""
    import random

    fillers = PT_BR_FILLERS if filler_type == "pt_br" else FR_PT_FILLERS
    results = []

    for s in sentences:
        words = s.split()
        if len(words) < 4:
            results.append({"original": s, "with_filler": s, "filler_type": filler_type})
            continue

        # Insert filler at ~40% of sentence
        pos = max(1, len(words) * 2 // 5)
        filler = random.choice(fillers)
        words.insert(pos, f"{filler}...")
        results.append({
            "original": s,
            "with_filler": " ".join(words),
            "filler_type": filler_type,
        })

    return results


def generate_french_portuguese_sentences(
    n_sentences: int = 500,
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict]:
    """Generate sentences typical of French speakers learning Portuguese.

    Based on Pipecat method: use LLM to generate diverse training text.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set — cannot generate French-PT sentences")
        return []

    client = anthropic.Anthropic(api_key=api_key)
    results = []

    contexts = [
        "reuniao de trabalho",
        "conversa informal com amigos",
        "apresentacao de projeto",
        "negociacao comercial",
        "aula de portugues",
        "pedindo informacoes na rua",
        "restaurante pedindo comida",
        "entrevista de emprego",
        "ligacao telefonica profissional",
        "discussao tecnica sobre software",
    ]

    for ctx in contexts:
        prompt = f"""Gere {n_sentences // len(contexts)} frases que um FRANCES de nivel B1-B2 diria em portugues durante: {ctx}

Inclua variedade:
- Frases COMPLETAS (pensamento terminado, pode traduzir)
- Frases INCOMPLETAS (vai continuar falando, nao traduzir)
- Com hesitacoes tipicas (euh, alors, como se diz, tipo)
- Com erros de conjugacao comuns de franceses
- Com code-switching involuntario (palavra em frances no meio)

Responda em JSON:
[{{"text": "frase", "label": "completo" ou "incompleto", "notes": "o que torna dificil classificar"}}]"""

        try:
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                batch = json.loads(json_match.group())
                for item in batch:
                    results.append({
                        "text": item["text"],
                        "label": item["label"],
                        "source": "claude_fr_pt",
                        "context": ctx,
                        "notes": item.get("notes", ""),
                    })
                log.info("  Generated %d sentences for context: %s", len(batch), ctx)

        except Exception as e:
            log.warning("Error generating for %s: %s", ctx, e)

        time.sleep(1)

    log.info("Total French-PT sentences generated: %d", len(results))
    return results


def run_full_pipeline(
    max_transcripts: int = 5000,
    max_fr_sentences: int = 500,
) -> Path:
    """Run the full label generation pipeline."""
    output_dir = DATA_DIR / "claude_labeled"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load transcripts
    log.info("=== Step 1: Loading Portuguese transcripts ===")
    mupe = load_coraa_transcripts(max_samples=max_transcripts)
    nurc = load_nurc_transcripts(max_samples=max_transcripts)
    all_transcripts = mupe + nurc
    log.info("Total transcripts: %d (MuPe=%d, NURC=%d)", len(all_transcripts), len(mupe), len(nurc))

    # Step 2: Claude classifies
    log.info("\n=== Step 2: Claude classifies complete/incomplete ===")
    sentences = [t["text"] for t in all_transcripts]
    classified = classify_with_claude(sentences)

    # Filter bad sentences (Pipecat removed 50-80%)
    kept = [c for c in classified if c["keep"]]
    removed = len(classified) - len(kept)
    log.info("Classified: %d total, %d kept, %d removed (%.0f%%)",
             len(classified), len(kept), removed, 100 * removed / max(len(classified), 1))

    complete = [c for c in kept if c["label"] == "completo"]
    incomplete = [c for c in kept if c["label"] == "incompleto"]
    log.info("  Complete: %d, Incomplete: %d", len(complete), len(incomplete))

    # Save classified
    with open(output_dir / "classified_pt.json", "w") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    # Step 3: Insert fillers (creates INCOMPLETE variants)
    log.info("\n=== Step 3: Insert fillers (PT-BR + FR-PT) ===")
    complete_texts = [c["text"] for c in complete[:1000]]

    pt_fillers = insert_fillers_with_claude(complete_texts, filler_type="pt_br")
    fr_fillers = insert_fillers_with_claude(complete_texts[:500], filler_type="fr_pt")

    with open(output_dir / "fillers_pt_br.json", "w") as f:
        json.dump(pt_fillers, f, indent=2, ensure_ascii=False)
    with open(output_dir / "fillers_fr_pt.json", "w") as f:
        json.dump(fr_fillers, f, indent=2, ensure_ascii=False)

    log.info("Fillers: %d PT-BR, %d FR-PT", len(pt_fillers), len(fr_fillers))

    # Step 4: Generate French-Portuguese sentences
    log.info("\n=== Step 4: Generate French-Portuguese sentences ===")
    fr_sentences = generate_french_portuguese_sentences(n_sentences=max_fr_sentences)
    with open(output_dir / "french_portuguese.json", "w") as f:
        json.dump(fr_sentences, f, indent=2, ensure_ascii=False)

    # Summary
    summary = {
        "total_transcripts": len(all_transcripts),
        "classified_kept": len(kept),
        "classified_removed": removed,
        "complete": len(complete),
        "incomplete": len(incomplete),
        "fillers_pt_br": len(pt_fillers),
        "fillers_fr_pt": len(fr_fillers),
        "french_portuguese": len(fr_sentences),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("\n=== Summary ===")
    for k, v in summary.items():
        log.info("  %s: %d", k, v)

    return output_dir


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    run_full_pipeline(max_transcripts=5000, max_fr_sentences=500)
