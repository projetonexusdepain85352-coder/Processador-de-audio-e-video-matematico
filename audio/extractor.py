# audio/extractor.py - Fase 1: Extracao de audio via ffmpeg
import subprocess, sys
from pathlib import Path
import config

def extrair_audio(video_path, audio_path=None):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video nao encontrado: {video_path}")
    if audio_path is None:
        audio_path = config.TEMP_DIR / "audio.wav"
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(config.AUDIO_SAMPLE_RATE),
        "-ac", str(config.AUDIO_CHANNELS),
        str(audio_path),
    ]
    print(f"[Fase 1] Extraindo audio de '{video_path.name}'...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg falhou:\n{r.stderr}")
    print(f"[Fase 1] Audio extraido: {audio_path}")
    return audio_path

def obter_info_video(video_path):
    video_path = Path(video_path)
    info = {}
    def ffprobe(*args):
        return subprocess.run(
            ["ffprobe", "-v", "error"] + list(args) + [str(video_path)],
            capture_output=True, text=True
        ).stdout.strip()

    try:
        info["duracao_s"] = float(ffprobe(
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1"
        ))
    except:
        info["duracao_s"] = 0.0

    saida = ffprobe(
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "default=noprint_wrappers=1"
    )
    for linha in saida.splitlines():
        if "=" not in linha:
            continue
        k, v = linha.split("=", 1)
        if k == "width":
            info["largura"] = int(v)
        elif k == "height":
            info["altura"] = int(v)
        elif k == "r_frame_rate":
            try:
                n, d = v.split("/")
                info["fps"] = round(int(n) / int(d), 2)
            except:
                info["fps"] = 30.0

    audio_saida = ffprobe(
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1"
    )
    info["tem_audio"] = bool(audio_saida)
    return info

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m audio.extractor <video.mp4>")
        sys.exit(1)
    print(obter_info_video(sys.argv[1]))
    extrair_audio(sys.argv[1])
