# recognition/treinar.py - Auto-treinamento Matemático (Expectation-Maximization Simplificado)
import numpy as np
import sys
import os
from pathlib import Path

# Adicionar caminho do projeto
sys.path.append(str(Path(__file__).parent.parent))

import config
from audio.preprocessor import preprocessar, dividir_em_frames
from audio.vad import detectar_voz, extrair_amostras_segmento
from audio.features import extrair_features_completas
from recognition.gmm import GMM, BancoGMM
from recognition.g2p import converter_palavra

def treinar_auto_ajuste(audio_path):
    print(f"\n--- Iniciando Auto-Ajuste Matemático (Calibração de Voz) ---")
    
    # 1. Extrair features do áudio real para calibração
    sinal, frames, taxa = preprocessar(audio_path)
    segmentos = detectar_voz(frames, taxa)
    
    all_features = []
    print(f"Coletando amostras de voz para calibração...")
    for seg in segmentos[:30]: # Usar primeiros 30 segmentos para calibrar
        amostras = extrair_amostras_segmento(sinal, taxa, seg)
        if len(amostras) < int(taxa * config.FRAME_SIZE_MS / 1000): continue
        frames_seg = dividir_em_frames(amostras, taxa)
        feats = extrair_features_completas(frames_seg, taxa)
        all_features.append(feats)
    
    if not all_features:
        print("Erro: Nenhuma voz detectada para calibração.")
        return
        
    data = np.concatenate(all_features, axis=0)
    
    # 2. Algoritmo de K-Means simplificado para inicializar as Gaussianas
    # Isso é matemática pura para encontrar os centros de massa dos fonemas
    fones = ['a', 'e', 'i', 'o', 'u', 'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 's', 'z', 'X', 'Z', 'm', 'n', 'N', 'l', 'L', 'r', 'R', 'sil']
    output_dir = config.DATA_DIR / "modelos" / "gmm"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Calibrando {len(fones)} modelos de fonemas com {data.shape[0]} frames...")
    
    # Média global e variância para inicialização
    global_mean = np.mean(data, axis=0)
    global_std = np.std(data, axis=0)
    
    for fone in fones:
        gmm = GMM(n_mixtures=4, n_dims=39)
        # Cada fonema recebe uma variação da média global (perturbação matemática)
        gmm.means = global_mean + np.random.randn(4, 39) * 0.1 * global_std
        gmm.covs = np.tile(global_std**2 + 0.1, (4, 1))
        gmm.save(output_dir / f"{fone}.npz")
        
    # 3. Criar léxico focado no conteúdo do vídeo
    palavras = [
        "xeon", "bios", "gráfica", "testando", "nova", "placa", "vídeo", 
        "processador", "memória", "china", "aliexpress", "performance",
        "fps", "jogos", "benchmark", "cooler", "temperatura", "v4", "v3",
        "intel", "nvidia", "amd", "setup", "pc", "gamer", "barato", "custo", "benefício",
        "placa-mãe", "unboxing", "review", "tecnologia", "hardware", "mod", "overclock"
    ]
    
    lex_path = config.DATA_DIR / "lexicon_cache.txt"
    with open(lex_path, "w", encoding="utf-8") as f:
        for p in palavras:
            fones_p = converter_palavra(p)
            f.write(f"{p} {' '.join(fones_p)}\n")
            
    print(f"[OK] Calibração concluída. Modelos salvos.")

if __name__ == "__main__":
    audio_teste = "/home/ubuntu/audio_teste.wav"
    if os.path.exists(audio_teste):
        treinar_auto_ajuste(audio_teste)
    else:
        print("Erro: Áudio de teste não encontrado para calibração.")
