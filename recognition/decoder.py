# recognition/decoder.py - Decodificador MFCC -> Texto
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import config
from recognition.gmm import BancoGMM
from recognition.hmm import SistemaHMM

@dataclass
class SegmentoDecodificado:
    inicio_s: float
    fim_s: float
    texto: str
    confianca: float

class Decodificador:
    def __init__(self, banco_gmm, sistema_hmm, dicionario):
        self.banco_gmm = banco_gmm
        self.sistema_hmm = sistema_hmm
        self.dicionario = dicionario

    def decodificar_segmento(self, features, inicio_s, fim_s):
        from recognition.hmm import decodificar_segmento_matematico
        
        if not self.banco_gmm.modelos:
            return SegmentoDecodificado(inicio_s, fim_s, "[inaudível]", 0.0)
        
        texto = decodificar_segmento_matematico(features, self.dicionario, self.banco_gmm, self.sistema_hmm)
        
        return SegmentoDecodificado(inicio_s, fim_s, texto, 1.0)

def carregar_modelos():
    banco = BancoGMM()
    if not banco.carregar_todos(config.DATA_DIR / "modelos" / "gmm"):
        return None
    
    fones = list(banco.modelos.keys())
    sistema = SistemaHMM(fones)
    
    # Carregar dicionário
    dicionario = {}
    lex_path = config.DATA_DIR / "lexicon_cache.txt"
    if lex_path.exists():
        with open(lex_path, "r", encoding="utf-8") as f:
            for linha in f:
                partes = linha.strip().split()
                if len(partes) > 1:
                    dicionario[partes[0]] = partes[1:]
                    
    return banco, sistema, dicionario

def salvar_transcricao(segmentos, path_txt, path_json):
    import json
    with open(path_txt, "w", encoding="utf-8") as f:
        for s in segmentos:
            f.write(f"[{s.inicio_s:.2f}s - {s.fim_s:.2f}s] {s.texto}\n")
            
    dados = [
        {"inicio": s.inicio_s, "fim": s.fim_s, "texto": s.texto, "conf": s.confianca}
        for s in segmentos
    ]
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)
