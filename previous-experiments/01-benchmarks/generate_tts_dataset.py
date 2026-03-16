"""
Generate Portuguese conversation dataset with real TTS speech.

Uses Microsoft Edge TTS to create dialogues between two speakers
with precise turn annotations. This produces real speech audio
that properly exercises Whisper-based turn-taking models.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
TTS_DIR = DATA_DIR / "portuguese_tts"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

SPEAKER_A_VOICE = "pt-BR-AntonioNeural"  # Male
SPEAKER_B_VOICE = "pt-BR-FranciscaNeural"  # Female

# Portuguese dialogue lines
DIALOGUES = [
    # Dialogue 1: Planning a trip
    [
        ("A", "Olha, eu estava pensando em viajar no próximo mês. Você tem alguma sugestão?"),
        ("B", "Que legal! Eu acho que o nordeste é uma ótima opção nessa época do ano."),
        ("A", "Verdade, eu nunca fui para Salvador. Dizem que a comida é incrível."),
        ("B", "É maravilhosa! O acarajé e a moqueca são imperdíveis."),
        ("A", "Quanto tempo você acha que eu preciso para conhecer a cidade?"),
        ("B", "Pelo menos uma semana. Tem muita coisa para ver e fazer."),
        ("A", "Entendi. Vou pesquisar preços de passagem então."),
        ("B", "Boa ideia! Se precisar de dicas de hotel, me fala."),
    ],
    # Dialogue 2: Work project
    [
        ("A", "Bom dia! Você viu o email sobre o novo projeto?"),
        ("B", "Vi sim. Parece bem interessante, mas o prazo está apertado."),
        ("A", "Concordo. Acho que precisamos dividir as tarefas logo."),
        ("B", "Eu posso ficar com a parte de pesquisa e documentação."),
        ("A", "Perfeito. Então eu fico com o desenvolvimento e os testes."),
        ("B", "Vamos marcar uma reunião amanhã para alinhar tudo?"),
        ("A", "Pode ser às dez da manhã?"),
        ("B", "Combinado. Vou mandar o convite agora."),
    ],
    # Dialogue 3: Weekend plans
    [
        ("A", "O que você vai fazer no fim de semana?"),
        ("B", "Estou pensando em ir ao cinema. Tem um filme novo que parece bom."),
        ("A", "Qual filme? Eu também estou querendo sair um pouco."),
        ("B", "É um drama brasileiro que está concorrendo a prêmios internacionais."),
        ("A", "Ah, eu ouvi falar! Dizem que a atuação é excelente."),
        ("B", "Vamos juntos então? A sessão das sete é boa?"),
        ("A", "Perfeito para mim. A gente pode jantar depois."),
        ("B", "Ótima ideia! Conheço um restaurante novo perto do cinema."),
    ],
    # Dialogue 4: Technical discussion
    [
        ("A", "Eu preciso de ajuda com um problema no código."),
        ("B", "Claro, o que está acontecendo?"),
        ("A", "A aplicação está travando quando tento processar arquivos grandes."),
        ("B", "Pode ser um problema de memória. Você está carregando tudo de uma vez?"),
        ("A", "Sim, eu leio o arquivo inteiro para a memória."),
        ("B", "Tenta usar streaming ou processar em pedaços menores."),
        ("A", "Faz sentido. Vou refatorar essa parte do código."),
        ("B", "Se precisar, eu tenho um exemplo que pode te ajudar."),
        ("A", "Seria ótimo! Pode me mandar por email?"),
        ("B", "Vou mandar agora mesmo. É bem simples de implementar."),
    ],
    # Dialogue 5: With holds (same speaker continues)
    [
        ("A", "Então, sobre aquele assunto que conversamos ontem."),
        ("A", "Eu pensei bastante e acho que devemos seguir em frente."),
        ("B", "Concordo totalmente. Na verdade, eu já comecei a preparar."),
        ("B", "Separei todos os documentos que vamos precisar."),
        ("A", "Excelente! Quando podemos começar?"),
        ("B", "Na próxima segunda-feira seria ideal."),
        ("A", "Segunda está perfeito."),
        ("A", "Vou avisar o resto da equipe sobre o plano."),
        ("B", "Boa. E eu confirmo com os fornecedores."),
    ],
    # Dialogue 6: Mixed holds and shifts
    [
        ("A", "Você ouviu a notícia sobre a empresa?"),
        ("B", "Não, o que aconteceu?"),
        ("A", "Eles vão abrir uma filial em Portugal."),
        ("A", "E estão procurando pessoas para transferir."),
        ("B", "Sério? Isso é muito interessante!"),
        ("B", "Eu sempre quis morar na Europa."),
        ("A", "Pois é, pode ser uma grande oportunidade."),
        ("B", "Vou me informar sobre os requisitos."),
        ("B", "Talvez eu me candidate para a vaga."),
        ("A", "Boa sorte! Eu acho que você tem boas chances."),
    ],
    # Dialogue 7: Academic discussion
    [
        ("A", "Como está indo a sua tese?"),
        ("B", "Estou na fase de análise de dados. É bem trabalhoso."),
        ("A", "Imagino. Qual é o tema mesmo?"),
        ("B", "Processamento de linguagem natural para o português."),
        ("B", "Especificamente, detecção de turnos em conversas."),
        ("A", "Que coincidência! Eu estou trabalhando em algo parecido."),
        ("A", "Meu foco é em modelos de tempo real."),
        ("B", "Que legal! Podemos trocar referências bibliográficas."),
        ("A", "Com certeza. Tenho alguns artigos muito bons sobre o assunto."),
        ("B", "Perfeito, vamos marcar um café para discutir."),
    ],
    # Dialogue 8: Shopping
    [
        ("A", "Preciso comprar um presente de aniversário."),
        ("B", "Para quem? Eu posso te ajudar a escolher."),
        ("A", "É para minha mãe. Ela faz setenta anos."),
        ("B", "Que especial! O que ela gosta?"),
        ("A", "Ela adora ler e cozinhar."),
        ("A", "Também gosta muito de música brasileira."),
        ("B", "Que tal um livro de receitas de um chef famoso?"),
        ("B", "Ou um disco de vinil de MPB clássica?"),
        ("A", "O disco de vinil é uma ideia genial!"),
        ("B", "Eu conheço uma loja que tem uma coleção incrível."),
    ],
    # Dialogue 9: Health
    [
        ("A", "Estou pensando em começar a fazer exercício."),
        ("B", "Isso é ótimo! O que você tem em mente?"),
        ("A", "Talvez corrida ou natação. O que você recomenda?"),
        ("B", "A natação é mais fácil para as articulações."),
        ("A", "Verdade. Tem uma piscina perto da minha casa."),
        ("B", "Perfeito! Você pode nadar três vezes por semana."),
        ("A", "Vou me inscrever amanhã. Obrigado pela sugestão."),
        ("B", "De nada! Depois me conta como foi."),
    ],
    # Dialogue 10: Technology
    [
        ("A", "Você já experimentou os novos modelos de inteligência artificial?"),
        ("B", "Sim! É impressionante como eles evoluíram."),
        ("A", "Eu estou usando para programação e tradução."),
        ("A", "A qualidade melhorou muito nos últimos meses."),
        ("B", "Concordo. Principalmente para línguas como o português."),
        ("B", "Antes era muito focado em inglês."),
        ("A", "Exatamente. Agora funciona muito bem em português."),
        ("B", "Ainda tem desafios com gírias e expressões regionais."),
        ("A", "É verdade. Mas já está muito bom para uso profissional."),
        ("B", "Com certeza. A tendência é só melhorar."),
    ],
]


async def _synthesize(text: str, voice: str, output_path: str) -> None:
    """Synthesize speech using Edge TTS."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


async def generate_tts_conversations() -> list[dict]:
    """Generate Portuguese dialogue audio with Edge TTS."""
    import edge_tts

    TTS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    conversations = []

    for dial_idx, dialogue in enumerate(DIALOGUES):
        log.info("Generating dialogue %d/%d...", dial_idx + 1, len(DIALOGUES))

        # Synthesize each turn
        turn_audios = []
        for turn_idx, (speaker, text) in enumerate(dialogue):
            voice = SPEAKER_A_VOICE if speaker == "A" else SPEAKER_B_VOICE
            tmp_path = str(TTS_DIR / f"tmp_d{dial_idx}_t{turn_idx}.mp3")

            await _synthesize(text, voice, tmp_path)

            # Convert to WAV 16kHz mono
            import subprocess
            wav_path = tmp_path.replace(".mp3", ".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True, check=True,
            )
            audio, sr = sf.read(wav_path)
            turn_audios.append((speaker, text, audio.astype(np.float32), sr))

            # Cleanup temp mp3
            os.remove(tmp_path)
            os.remove(wav_path)

        # Concatenate with realistic gaps
        rng = np.random.default_rng(42 + dial_idx)
        sr = 16000
        chunks = []
        turns = []
        t = 0.0

        for i, (speaker, text, audio, _) in enumerate(turn_audios):
            # Gap between turns
            if i > 0:
                prev_speaker = dialogue[i - 1][0]
                if speaker != prev_speaker:
                    gap = rng.uniform(0.15, 0.5)  # Shift: longer gap
                else:
                    gap = rng.uniform(0.05, 0.2)  # Hold: short pause
                gap_samples = int(gap * sr)
                chunks.append(np.zeros(gap_samples, dtype=np.float32))
                t += gap

            start = t
            duration = len(audio) / sr
            end = start + duration

            turns.append({
                "speaker": speaker,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            })
            chunks.append(audio)
            t = end

        full_audio = np.concatenate(chunks)
        audio_path = TTS_DIR / f"pt_tts_dialogue_{dial_idx:03d}.wav"
        sf.write(str(audio_path), full_audio, sr)

        # Compute events
        turn_shifts = []
        holds = []
        for k in range(1, len(turns)):
            if turns[k]["speaker"] != turns[k - 1]["speaker"]:
                turn_shifts.append(turns[k]["start"])
            else:
                holds.append(turns[k]["start"])

        conv = {
            "conv_id": f"pt_tts_{dial_idx:03d}",
            "audio_path": str(audio_path),
            "sample_rate": sr,
            "duration": full_audio.shape[0] / sr,
            "turns": turns,
            "turn_shifts": turn_shifts,
            "holds": holds,
            "n_turns": len(turns),
            "n_turn_shifts": len(turn_shifts),
            "n_holds": len(holds),
        }
        conversations.append(conv)
        log.info("  Dialogue %d: %d turns, %d shifts, %d holds, %.1fs",
                 dial_idx, len(turns), len(turn_shifts), len(holds),
                 full_audio.shape[0] / sr)

    # Save annotations
    ann_path = ANNOTATIONS_DIR / "portuguese_tts_annotations.json"
    with open(ann_path, "w") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    log.info("Saved %d TTS dialogue annotations to %s", len(conversations), ann_path)

    return conversations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(generate_tts_conversations())
