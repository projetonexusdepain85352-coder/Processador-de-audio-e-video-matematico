# recognition/hmm.py - Hidden Markov Model + Viterbi de Alta Precisão
import numpy as np

class HMM:
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.trans = np.zeros((n_states, n_states))
        for i in range(n_states):
            self.trans[i, i] = 0.6 
            if i < n_states - 1:
                self.trans[i, i+1] = 0.4
        self.start_prob = np.zeros(n_states)
        self.start_prob[0] = 1.0

def viterbi_log(obs, hmm, gmm):
    T = obs.shape[0]
    N = hmm.n_states
    v = np.ones((T, N)) * -np.inf
    
    log_emissions = gmm.log_likelihood(obs)
    
    v[0, 0] = np.log(hmm.start_prob[0] + 1e-10) + log_emissions[0]
    
    for t in range(1, T):
        for j in range(N):
            # Transições: i -> j
            prev_probs = v[t-1, :] + np.log(hmm.trans[:, j] + 1e-10)
            v[t, j] = np.max(prev_probs) + log_emissions[t]
                        
    return np.max(v[T-1, :])

def decodificar_segmento_matematico(features, lexicon, banco_gmm, sistema_hmm):
    if not banco_gmm.modelos:
        return "[inaudível]"

    melhor_palavra = "[inaudível]"
    melhor_score = -np.inf
    
    # Penalidade por frame para normalizar palavras de tamanhos diferentes
    # Isso impede que palavras curtas ganhem apenas por terem menos frames
    for palavra, fones in lexicon.items():
        if len(palavra) < 3: continue # Ignorar palavras muito curtas para evitar ruído
        
        n_fones = len(fones)
        frames_por_fone = features.shape[0] // n_fones
        if frames_por_fone < 2: continue # Segmento muito curto para esta palavra
        
        score_total = 0
        for i, fone in enumerate(fones):
            if fone in banco_gmm.modelos:
                start = i * frames_por_fone
                end = (i + 1) * frames_por_fone if i < n_fones - 1 else features.shape[0]
                chunk = features[start:end]
                
                score_fone = viterbi_log(chunk, sistema_hmm.modelos.get(fone, HMM()), banco_gmm.modelos[fone])
                score_total += score_fone
        
        # Normalização Bayesiana: Score médio por frame + bônus por extensão de palavra
        # Matemática pura para compensar a variância temporal
        score_final = (score_total / features.shape[0]) + (0.1 * len(palavra))
        
        if score_final > melhor_score:
            melhor_score = score_final
            melhor_palavra = palavra
            
    # Limiar de confiança estrito para evitar alucinação
    # Se a probabilidade for muito baixa, o sistema admite que não sabe
    return melhor_palavra if melhor_score > -50 else "[inaudível]"

class SistemaHMM:
    def __init__(self, fones):
        self.fones = fones
        self.modelos = {f: HMM() for f in fones}
