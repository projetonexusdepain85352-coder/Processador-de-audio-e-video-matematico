# audio/preprocessor.py - Fase 2: Pre-processamento do sinal
import numpy as np
import wave
import sys
from pathlib import Path
import config

def carregar_wav(audio_path):
    with wave.open(str(audio_path), "rb") as wf:
        n_canais  = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames  = wf.getnframes()
        raw       = wf.readframes(n_frames)
    if n_canais != 1:
        raise ValueError(f"Esperado mono (1 canal), encontrado {n_canais}.")
    if sampwidth != 2:
        raise ValueError(f"Esperado PCM 16-bit, encontrado {sampwidth} bytes.")
    sinal = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    sinal /= 32768.0
    print(f"[Fase 2] WAV: {len(sinal)/framerate:.2f}s | {framerate}Hz")
    return sinal, framerate

def remover_dc_offset(sinal):
    return sinal - np.mean(sinal)

def normalizar(sinal):
    m = np.max(np.abs(sinal))
    return sinal if m == 0 else sinal / m

def aplicar_pre_enfase(sinal, coef=config.PRE_EMPHASIS):
    return np.append(sinal[0], sinal[1:] - coef * sinal[:-1])

def dividir_em_frames(sinal, taxa,
                      frame_ms=config.FRAME_SIZE_MS,
                      passo_ms=config.FRAME_STEP_MS):
    frame_len = int(taxa * frame_ms / 1000)
    passo_len = int(taxa * passo_ms / 1000)
    n_frames  = 1 + (len(sinal) - frame_len) // passo_len
    if n_frames <= 0:
        raise ValueError("Sinal muito curto para o tamanho de frame.")
    indices = (
        np.arange(frame_len)[np.newaxis, :]
        + np.arange(n_frames)[:, np.newaxis] * passo_len
    )
    indices = np.clip(indices, 0, len(sinal) - 1)
    return sinal[indices] * np.hamming(frame_len)

def preprocessar(audio_path):
    sinal, taxa = carregar_wav(audio_path)
    print("[Fase 2] Aplicando pre-processamento...")
    sinal = remover_dc_offset(sinal)
    sinal = normalizar(sinal)
    sinal = aplicar_pre_enfase(sinal)
    frames = dividir_em_frames(sinal, taxa)
    print(f"[Fase 2] {len(frames)} frames gerados.")
    return sinal, frames, taxa

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m audio.preprocessor <audio.wav>")
        sys.exit(1)
    sinal, frames, taxa = preprocessar(sys.argv[1])
    print(f"Frames: {frames.shape}")
