# recognition/g2p.py - Conversor Grafema -> Fonema (Regras PT-BR)
import re

# Mapeamento simplificado de fonemas para PT-BR (baseado no FalaBrasil)
# Vogais: a, e, i, o, u, E (pé), O (pó), @ (nasal)
# Consoantes: p, b, t, d, k, g, f, v, s, z, X (ch), Z (j), m, n, N (nh), l, L (lh), r, R (rr)

REGRAS = [
    (r'ch', 'X'), (r'lh', 'L'), (r'nh', 'N'),
    (r'rr', 'R'), (r'ss', 's'), (r'qu', 'k'),
    (r'gu', 'g'), (r'ce', 'se'), (r'ci', 'si'),
    (r'ca', 'ka'), (r'co', 'ko'), (r'cu', 'ku'),
    (r'ge', 'Ze'), (r'gi', 'Zi'), (r'ga', 'ga'),
    (r'go', 'go'), (r'gu', 'gu'),
    (r'x', 'X'), (r'j', 'Z'), (r'h', ''),
    (r'y', 'i'), (r'w', 'u'),
]

def converter_palavra(palavra):
    p = palavra.lower().strip()
    if not p: return []
    
    # Aplicar regras de substituição
    for padrao, sub in REGRAS:
        p = re.sub(padrao, sub, p)
    
    # Mapeamento direto de caracteres restantes
    fones = []
    for c in p:
        if c in 'abcdefghijklmnopqrstuvwxyzXRLNZZeEiIoOuU@':
            fones.append(c)
    
    return fones

def obter_lexicon(palavras):
    lexicon = {}
    for p in palavras:
        lexicon[p] = converter_palavra(p)
    return lexicon

if __name__ == "__main__":
    testes = ["casa", "cachorro", "amanhã", "xeon", "vídeo"]
    for t in testes:
        print(f"{t} -> {' '.join(converter_palavra(t))}")
