# audio/features.py - Fase 4: Extracao de MFCCs
import numpy as np
import sys
import config

def hz_para_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_para_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

_BANCO = None
_N_FFT = 512

def criar_banco_filtros_mel(n_filtros=config.MFCC_N_FILTERS, n_fft=512,
                             taxa=config.AUDIO_SAMPLE_RATE,
                             freq_min=config.MFCC_LOW_FREQ_HZ,
                             freq_max=config.MFCC_HIGH_FREQ_HZ):
    freq_max  = min(freq_max, taxa / 2)
    pts_mel   = np.linspace(hz_para_mel(freq_min), hz_para_mel(freq_max), n_filtros + 2)
    pts_hz    = mel_para_hz(pts_mel)
    n_bins    = n_fft // 2 + 1
    bins      = np.clip(np.floor((n_fft + 1) * pts_hz / taxa).astype(int), 0, n_bins - 1)
    banco     = np.zeros((n_filtros, n_bins))
    for m in range(1, n_filtros + 1):
        fl, fc, fr = bins[m-1], bins[m], bins[m+1]
        for k in range(fl, fc):
            if fc > fl:
                banco[m-1, k] = (k - fl) / (fc - fl)
        for k in range(fc, fr):
            if fr > fc:
                banco[m-1, k] = (fr - k) / (fr - fc)
        banco[m-1, fc] = 1.0
    return banco

def _get_banco(taxa):
    global _BANCO
    if _BANCO is None:
        _BANCO = criar_banco_filtros_mel(n_fft=_N_FFT, taxa=taxa)
    return _BANCO

def dct_tipo2(x):
    N = len(x)
    n = np.arange(N)
    k = np.arange(N)[:, np.newaxis]
    r = np.dot(np.cos(np.pi * k * (2 * n + 1) / (2 * N)), x)
    r[0]  *= np.sqrt(1.0 / (4 * N))
    r[1:] *= np.sqrt(1.0 / (2 * N))
    return r

def calcular_mfcc_frames(frames, taxa, n_coeffs=config.MFCC_N_COEFFS):
    banco = _get_banco(taxa)
    mfccs = np.zeros((len(frames), n_coeffs))
    for i, frame in enumerate(frames):
        pot  = (np.abs(np.fft.rfft(frame, n=_N_FFT)) ** 2) / _N_FFT
        emel = np.dot(banco, pot)
        emel = np.where(emel > 1e-10, emel, 1e-10)
        mfccs[i] = dct_tipo2(np.log(emel))[:n_coeffs]
    return mfccs

def calcular_deltas(f, N=2):
    nf, nc = f.shape
    d   = np.zeros_like(f)
    den = 2 * sum(n ** 2 for n in range(1, N + 1))
    for t in range(nf):
        num = np.zeros(nc)
        for n in range(1, N + 1):
            num += n * (f[min(t+n, nf-1)] - f[max(t-n, 0)])
        d[t] = num / den
    return d

def extrair_features_completas(frames, taxa, n_coeffs=config.MFCC_N_COEFFS):
    m  = calcular_mfcc_frames(frames, taxa, n_coeffs)
    
    # CMVN - Cepstral Mean and Variance Normalization (Matemática Pura para Precisão)
    # Isso remove o efeito do canal/microfone e foca no fonema
    m = (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 1e-10)
    
    d1 = calcular_deltas(m)
    d2 = calcular_deltas(d1)
    feat = np.concatenate([m, d1, d2], axis=1)
    return feat

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m audio.features <audio.wav>")
        sys.exit(1)
    sys.path.insert(0, ".")
    from audio.preprocessor import preprocessar
    sinal, frames, taxa = preprocessar(sys.argv[1])
    feat = extrair_features_completas(frames, taxa)
    np.save("temp/features.npy", feat)
    print("Salvo em temp/features.npy")
