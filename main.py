# main.py
# Orquestrador principal — executa as fases disponíveis
#
# Uso:
#   python main.py video.mp4                      # roda tudo
#   python main.py video.mp4 --fase audio         # só extração de áudio
#   python main.py video.mp4 --fase vad           # até detecção de voz
#   python main.py video.mp4 --fase features      # até MFCCs
#   python main.py video.mp4 --fase transcricao   # até transcrição de áudio (sem OCR)
#   python main.py video.mp4 --fase ocr           # só OCR de tela
#   python main.py video.mp4 --fase tudo          # tudo (padrão)

import argparse
import json
import sys
import time
from pathlib import Path

import config


def parse_args():
    p = argparse.ArgumentParser(
        description="Processador de videos do Instagram - sem IA"
    )
    p.add_argument("video", help="Caminho para o video .mp4")
    p.add_argument(
        "--fase",
        choices=["audio", "vad", "features", "transcricao", "ocr", "tudo"],
        default="transcricao",
        help="Ate qual fase executar (padrao: transcricao)",
    )
    p.add_argument(
        "--fps-ocr",
        type=float,
        default=config.OCR_FRAMES_PER_SEC,
        help=f"Frames por segundo para OCR (padrao: {config.OCR_FRAMES_PER_SEC})",
    )
    p.add_argument(
        "--salvar-features",
        action="store_true",
        help="Salvar features MFCC em arquivo .npy",
    )
    return p.parse_args()


def executar(args):
    import numpy as np
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERRO] Video nao encontrado: {video_path}")
        sys.exit(1)

    nome_base = video_path.stem
    resultado = {"video": str(video_path), "fases": {}}
    inicio_total = time.time()

    print(f"\n{'='*55}")
    print(f"  Processando: {video_path.name}")
    print(f"{'='*55}\n")

    # ─────────────────────────────────────────────
    # FASE 1 — Extração de Áudio
    # ─────────────────────────────────────────────
    from audio.extractor import extrair_audio, obter_info_video
    from audio.preprocessor import preprocessar, dividir_em_frames
    from audio.vad import detectar_voz, extrair_amostras_segmento
    from audio.features import extrair_features_completas

    info_video = obter_info_video(str(video_path))
    dur = info_video.get("duracao_s", 0) or 0
    print(f"[Info] {dur:.1f}s | "
          f"{info_video.get('largura','?')}x{info_video.get('altura','?')} | "
          f"{info_video.get('fps','?')} fps\n")

    audio_path = None
    if not info_video.get("tem_audio"):
        print("[AVISO] Sem audio - pulando fases de audio.")
        if args.fase not in ("ocr",):
            args.fase = "ocr"
    else:
        t = time.time()
        audio_path = config.TEMP_DIR / f"{nome_base}.wav"
        audio_path = extrair_audio(str(video_path), audio_path)
        resultado["fases"]["audio"] = {
            "arquivo": str(audio_path),
            "tempo_s": round(time.time() - t, 2),
        }
        print()

    if args.fase == "audio":
        _finalizar(resultado, inicio_total)
        return resultado

    # ─────────────────────────────────────────────
    # FASE 2 — Pré-processamento
    # ─────────────────────────────────────────────
    t = time.time()
    sinal, frames, taxa = preprocessar(audio_path)
    resultado["fases"]["preprocessamento"] = {
        "n_frames": len(frames),
        "taxa_hz": taxa,
        "tempo_s": round(time.time() - t, 2),
    }
    print()

    # ─────────────────────────────────────────────
    # FASE 3 — VAD
    # ─────────────────────────────────────────────
    t = time.time()
    segmentos = detectar_voz(frames, taxa)
    resultado["fases"]["vad"] = {
        "n_segmentos": len(segmentos),
        "segmentos": [
            {"inicio_s": s.inicio_s, "fim_s": s.fim_s, "duracao_ms": s.duracao_ms}
            for s in segmentos
        ],
        "tempo_s": round(time.time() - t, 2),
    }
    print()

    if args.fase == "vad":
        _finalizar(resultado, inicio_total)
        return resultado

    # ─────────────────────────────────────────────
    # FASE 4 — MFCCs
    # ─────────────────────────────────────────────
    t = time.time()
    print(f"[Fase 4] Extraindo MFCCs de {len(segmentos)} segmentos...")
    features_por_segmento = []

    for i, seg in enumerate(segmentos):
        amostras = extrair_amostras_segmento(sinal, taxa, seg)
        if len(amostras) < int(taxa * config.FRAME_SIZE_MS / 1000):
            continue
        frames_seg = dividir_em_frames(amostras, taxa)
        feats = extrair_features_completas(frames_seg, taxa)
        features_por_segmento.append({
            "segmento_idx": i,
            "inicio_s": seg.inicio_s,
            "fim_s": seg.fim_s,
            "n_frames": len(feats),
            "features": feats,
        })
        # Progress a cada 50 segmentos
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(segmentos)} segmentos processados...")

    resultado["fases"]["features"] = {
        "n_segmentos_processados": len(features_por_segmento),
        "dimensoes": 39,
        "tempo_s": round(time.time() - t, 2),
    }

    if args.salvar_features:
        for item in features_por_segmento:
            idx = item["segmento_idx"]
            np.save(config.TEMP_DIR / f"features_seg_{idx:04d}.npy", item["features"])

    print(f"[Fase 4] {len(features_por_segmento)} segmentos com features.\n")

    if args.fase == "features":
        _finalizar(resultado, inicio_total)
        return resultado

    # ─────────────────────────────────────────────
    # FASE 5/6 — DECODIFICAÇÃO (transcrição de áudio)
    # ─────────────────────────────────────────────
    if args.fase in ("transcricao", "tudo"):
        from recognition.decoder import carregar_modelos, Decodificador, salvar_transcricao

        modelos = carregar_modelos()
        if modelos is None:
            print("[AVISO] Modelos nao encontrados. Execute primeiro:")
            print("        python -m recognition.treinar")
            print("        Pulando transcricao de audio.\n")
        else:
            banco_gmm, sistema_hmm, dicionario = modelos
            decoder = Decodificador(banco_gmm, sistema_hmm, dicionario)

            print(f"[Fase 5/6] Decodificando {len(features_por_segmento)} segmentos...")
            t = time.time()

            segmentos_decoded = []
            for i, item in enumerate(features_por_segmento):
                seg = decoder.decodificar_segmento(
                    item["features"],
                    item["inicio_s"],
                    item["fim_s"],
                )
                segmentos_decoded.append(seg)
                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{len(features_por_segmento)} decodificados...")

            # Salvar transcrição
            saida_txt  = config.OUTPUT_DIR / f"{nome_base}_transcricao.txt"
            saida_json = config.OUTPUT_DIR / f"{nome_base}_transcricao.json"
            salvar_transcricao(segmentos_decoded, saida_txt, saida_json)

            n_com_texto = sum(1 for s in segmentos_decoded if s.texto.strip()
                              and "[inaudível]" not in s.texto)

            resultado["fases"]["transcricao"] = {
                "n_segmentos": len(segmentos_decoded),
                "n_com_texto": n_com_texto,
                "arquivo_txt": str(saida_txt),
                "arquivo_json": str(saida_json),
                "tempo_s": round(time.time() - t, 2),
            }
            print()

        if args.fase == "transcricao":
            _salvar_resultado(resultado, config.OUTPUT_DIR / f"{nome_base}_resultado.json")
            _finalizar(resultado, inicio_total)
            return resultado

    # ─────────────────────────────────────────────
    # FASE 7 — OCR
    # ─────────────────────────────────────────────
    if args.fase in ("ocr", "tudo"):
        from ocr.text_extractor import processar_video_ocr, salvar_resultado_ocr

        t = time.time()
        textos = processar_video_ocr(str(video_path), fps_analise=args.fps_ocr)

        saida_ocr = config.OUTPUT_DIR / f"{nome_base}_textos_tela.txt"
        salvar_resultado_ocr(textos, saida_ocr)

        resultado["fases"]["ocr"] = {
            "n_textos": len(textos),
            "textos": [
                {
                    "timestamp_s": tx.timestamp_s,
                    "texto": tx.texto,
                    "confianca": round(tx.confianca, 1),
                }
                for tx in textos
            ],
            "tempo_s": round(time.time() - t, 2),
        }
        print()

    # Salvar resultado final
    _salvar_resultado(resultado, config.OUTPUT_DIR / f"{nome_base}_resultado.json")
    _finalizar(resultado, inicio_total)
    return resultado


def _salvar_resultado(resultado: dict, caminho: Path):
    import numpy as np

    def serializar(obj):
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        raise TypeError(f"Nao serializavel: {type(obj)}")

    caminho.parent.mkdir(parents=True, exist_ok=True)
    resultado_limpo = json.loads(json.dumps(resultado, default=serializar))
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(resultado_limpo, f, ensure_ascii=False, indent=2)
    print(f"[OK] Resultado salvo em: {caminho}")


def _finalizar(resultado: dict, inicio: float):
    total = time.time() - inicio
    print(f"\n{'='*55}")
    print(f"  Concluido em {total:.1f}s")
    for fase, dados in resultado.get("fases", {}).items():
        print(f"  {fase}: {dados.get('tempo_s','?')}s")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    args = parse_args()
    executar(args)
