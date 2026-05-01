# audio/vad.py - Fase 3: Deteccao de Voz (VAD)
import numpy as np
import sys
from dataclasses import dataclass
import config

@dataclass
class SegmentoVoz:
    inicio_s:  float
    fim_s:     float
    inicio_ms: int
    fim_ms:    int

    @property
    def duracao_ms(self):
        return self.fim_ms - self.inicio_ms

    def __repr__(self):
        return f"Voz({self.inicio_ms}ms -> {self.fim_ms}ms, {self.duracao_ms}ms)"

def calcular_energia(frames):
    return np.sqrt(np.mean(frames ** 2, axis=1))

def calcular_zcr(frames):
    trocas = np.diff(np.sign(frames), axis=1)
    return np.sum(np.abs(trocas) > 0, axis=1) / frames.shape[1]

def _suavizar(mascara, janela=3):
    r = mascara.copy().astype(float)
    for i in range(janela, len(mascara) - janela):
        r[i] = float(np.mean(mascara[i - janela: i + janela + 1]) > 0.4)
    return r.astype(bool)

def _para_segmentos(mascara, passo_ms):
    segs = []; em = False; ini = 0
    for i, f in enumerate(mascara):
        ms = i * passo_ms
        if f and not em:
            ini = ms; em = True
        elif not f and em:
            segs.append((ini, ms)); em = False
    if em:
        segs.append((ini, len(mascara) * passo_ms))
    return segs

def _refinar(segs, pad_ms, min_fala, min_sil, total_ms):
    if not segs:
        return []
    com_pad = [(max(0, i - pad_ms), min(total_ms, f + pad_ms)) for i, f in segs]
    mes = [com_pad[0]]
    for ini, fim in com_pad[1:]:
        ui, uf = mes[-1]
        if ini - uf < min_sil:
            mes[-1] = (ui, fim)
        else:
            mes.append((ini, fim))
    return [
        SegmentoVoz(i / 1000, f / 1000, i, f)
        for i, f in mes if f - i >= min_fala
    ]

def detectar_voz(frames, taxa,
                 limiar_energia=config.VAD_ENERGY_THRESHOLD,
                 limiar_zcr=config.VAD_ZCR_THRESHOLD,
                 pad_ms=config.VAD_SPEECH_PAD_MS,
                 min_fala_ms=config.VAD_MIN_SPEECH_MS,
                 min_silencio_ms=config.VAD_MIN_SILENCE_MS):
    passo_ms = config.FRAME_STEP_MS
    energia  = calcular_energia(frames)
    zcr      = calcular_zcr(frames)
    emax     = np.percentile(energia, 99)
    enorm    = energia / emax if emax > 0 else energia

    eh_fala = (
        (enorm > limiar_energia) & (zcr < limiar_zcr) |
        (enorm > limiar_energia * 2) |
        (enorm > limiar_energia * 0.5) & (zcr > limiar_zcr)
    )
    eh_fala = _suavizar(eh_fala)
    segs = _refinar(
        _para_segmentos(eh_fala, passo_ms),
        pad_ms, min_fala_ms, min_silencio_ms, len(frames) * passo_ms
    )
    print(f"[Fase 3] {len(segs)} segmentos de fala detectados.")
    return segs

def extrair_amostras_segmento(sinal, taxa, seg):
    return sinal[int(seg.inicio_s * taxa): int(seg.fim_s * taxa)]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m audio.vad <audio.wav>")
        sys.exit(1)
    sys.path.insert(0, ".")
    from audio.preprocessor import preprocessar
    sinal, frames, taxa = preprocessar(sys.argv[1])
    for i, s in enumerate(detectar_voz(frames, taxa), 1):
        print(f"  {i}. {s}")
