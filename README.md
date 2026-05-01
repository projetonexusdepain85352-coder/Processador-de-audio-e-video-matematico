# Processador de Áudio e Vídeo

Script pessoal feito no tempo livre. Sem suporte. Sem garantia.

**Objetivo:** transcrever áudio de vídeos localmente, sem LLM, sem nuvem, usando métodos clássicos de processamento de sinais (MFCC + GMM + HMM). O OCR de texto em tela é secundário.

**Status atual:** pipeline de áudio até extração de features funciona. A transcrição roda mas não produz resultado útil ainda — veja detalhes abaixo.

---

## Arquitetura

```
Processador de audio e video/
├── audio/
│   ├── extractor.py        # Extração de áudio PCM via FFmpeg
│   ├── preprocessor.py     # Normalização, pré-ênfase, janelamento Hamming
│   ├── vad.py              # Detecção de voz por energia + ZCR
│   └── features.py         # MFCCs (39 dimensões) + CMVN
├── recognition/
│   ├── g2p.py              # Conversor grafema -> fonema (PT-BR)
│   ├── gmm.py              # Gaussian Mixture Model
│   ├── hmm.py              # HMM + algoritmo de Viterbi
│   ├── treinar.py          # Calibração dos modelos
│   └── decoder.py          # Orquestrador da transcrição
├── data/
│   ├── modelos/            # Parâmetros GMM salvos (.npz)
│   └── lexicon_cache.txt   # Léxico fonético
├── temp/                   # Áudio intermediário (WAV PCM)
├── output/                 # Resultados em JSON e TXT
├── config.py               # Parâmetros centrais
└── main.py                 # Ponto de entrada
```

---

## O que funciona

**Fase 1 — Extração de áudio:** FFmpeg extrai WAV mono 16kHz corretamente.

**Fase 2 — Pré-processamento:** DC offset, normalização, pré-ênfase (0.97) e janelamento Hamming implementados corretamente.

**Fase 3 — VAD:** detecta segmentos de fala por energia RMS + ZCR, com suavização e padding. Funciona.

**Fase 4 — MFCCs:** 13 coeficientes + delta + delta-delta (39 dimensões) com CMVN. É a parte mais sólida do projeto — a matemática está correta.

---

## O que não funciona

**OCR (fase 7):** `ocr/text_extractor.py` não existe. O `main.py` importa o módulo mas ele nunca foi implementado. Quebra em runtime. O caminho do Tesseract também está hardcoded para Windows no `config.py`.

**Transcrição de áudio (fases 5/6):** tecnicamente roda após calibração, mas não produz resultado útil:

- O léxico está hardcoded com ~35 palavras sobre hardware (`xeon`, `nvidia`, `aliexpress`, `overclock`...). Foi feito pra um vídeo específico — não é um transcritor genérico.
- O `treinar.py` não treina de verdade. Ele pega a média global de todos os frames e adiciona ruído aleatório (`np.random.randn * 0.1`) pra diferenciar os fonemas. Os modelos resultantes não representam fonemas reais.
- As probabilidades de transição do HMM são hardcoded (`0.6` self-loop, `0.4` forward) — não foram aprendidas de nenhum dado.
- A confiança retornada pelo decoder é sempre `1.0` — não é calculada.
- O limiar anti-alucinação (`-50`) e o bônus por extensão de palavra (`0.1 * len(palavra)`) são arbitrários.

Na prática: retorna uma das 35 palavras do léxico ou `[inaudível]`, independente do áudio.

---

## O que falta para funcionar de verdade

**Dados — gargalo principal.** GMM+HMM é uma arquitetura válida (é o que HTK e Sphinx usam), mas requer corpus acústico rotulado por fonema em PT-BR. Sem dados, nenhuma melhoria de código resolve. Opções:

- [FalaBrasil](http://www.laps.ufpa.br/falabrasil/) — corpus PT-BR com alinhamento fonético
- Gravar e rotular manualmente

**Código faltando:**

- `ocr/text_extractor.py` — precisa ser implementado (Tesseract + pytesseract)
- Treinamento EM real no `treinar.py` — Expectation-Maximization iterativo por fonema, ajustando médias, covariâncias e pesos dos GMMs com dados reais
- Léxico fonético completo do PT-BR — existe pronto: [Unitex-PB](https://unitexgramlab.org/)
- Beam search no decoder — o método atual é O(n_palavras) por segmento, não escala pra vocabulário real
- Cálculo real de confiança no decoder

---

## Como usar

```bash
python main.py video.mp4 --fase audio       # só extração de áudio
python main.py video.mp4 --fase vad         # até detecção de voz
python main.py video.mp4 --fase features    # até MFCCs — recomendado por ora
```

Para tentar a transcrição (com as limitações acima):

```bash
python -m recognition.treinar
python main.py video.mp4 --fase transcricao
```

---

## Requisitos

- Python 3.8+ (recomendado 3.11/3.12)
- FFmpeg no PATH
- numpy

---

## Licença

Domínio público. Faz o que quiser.
