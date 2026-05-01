# config.py - Configuracoes centrais do projeto
from pathlib import Path

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR   = BASE_DIR / "temp"
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

AUDIO_SAMPLE_RATE    = 16000
AUDIO_CHANNELS       = 1
AUDIO_BIT_DEPTH      = 16
PRE_EMPHASIS         = 0.97
FRAME_SIZE_MS        = 25
FRAME_STEP_MS        = 10
VAD_ENERGY_THRESHOLD = 0.02
VAD_ZCR_THRESHOLD    = 0.15
VAD_SPEECH_PAD_MS    = 50
VAD_MIN_SPEECH_MS    = 100
VAD_MIN_SILENCE_MS   = 200
MFCC_N_FILTERS       = 26
MFCC_N_COEFFS        = 13
MFCC_LOW_FREQ_HZ     = 80
MFCC_HIGH_FREQ_HZ    = 7600
OCR_FRAMES_PER_SEC   = 1
OCR_MIN_CONFIDENCE   = 60
OCR_LANG             = "por+eng"
OCR_DEDUP_THRESHOLD  = 0.85
OCR_MIN_TEXT_LENGTH  = 3
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
