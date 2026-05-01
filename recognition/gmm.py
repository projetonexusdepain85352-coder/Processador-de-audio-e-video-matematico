# recognition/gmm.py - Gaussian Mixture Model para Fonemas
import numpy as np

class GMM:
    def __init__(self, n_mixtures=4, n_dims=39):
        self.n_mixtures = n_mixtures
        self.n_dims = n_dims
        self.weights = np.ones(n_mixtures) / n_mixtures
        self.means = np.zeros((n_mixtures, n_dims))
        self.covs = np.ones((n_mixtures, n_dims)) # Diagonal covariance

    def log_likelihood(self, data):
        """Calcula log-verossimilhança dos dados dado o modelo GMM"""
        # data: (n_frames, n_dims)
        n_frames = data.shape[0]
        log_probs = np.zeros((n_frames, self.n_mixtures))
        
        for m in range(self.n_mixtures):
            diff = data - self.means[m]
            # Gaussian log-pdf (diagonal covariance)
            log_p = -0.5 * (self.n_dims * np.log(2 * np.pi) + 
                           np.sum(np.log(self.covs[m])) + 
                           np.sum((diff**2) / self.covs[m], axis=1))
            log_probs[:, m] = np.log(self.weights[m] + 1e-10) + log_p
            
        # Log-sum-exp trick
        max_log = np.max(log_probs, axis=1)
        return max_log + np.log(np.sum(np.exp(log_probs - max_log[:, np.newaxis]), axis=1))

    def save(self, path):
        np.savez(path, weights=self.weights, means=self.means, covs=self.covs)

    def load(self, path):
        data = np.load(path)
        self.weights = data['weights']
        self.means = data['means']
        self.covs = data['covs']
        self.n_mixtures = len(self.weights)
        self.n_dims = self.means.shape[1]

class BancoGMM:
    def __init__(self):
        self.modelos = {} # fone -> GMM

    def adicionar_modelo(self, fone, gmm):
        self.modelos[fone] = gmm

    def carregar_todos(self, pasta):
        from pathlib import Path
        for p in Path(pasta).glob("*.npz"):
            fone = p.stem
            gmm = GMM()
            gmm.load(p)
            self.modelos[fone] = gmm
        return len(self.modelos) > 0
