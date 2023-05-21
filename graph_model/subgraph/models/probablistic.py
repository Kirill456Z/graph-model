import imp
from minicache import cached_step

from scipy.spatial.distance import cdist
import numpy.typing as npt
import numpy as np


@cached_step()
def pairwise_distances(utterance_embeddings_1: np.array, utterance_embeddings_2: np.array):
    return cdist(utterance_embeddings_1, utterance_embeddings_2)


class ProbablisticClustering:
    def __init__(self, n_clusters):
        self.fit_ = False
        self.n_clusters = n_clusters

    def fit(self, cluster_centers: np.array, utterance_embeddings: np.array, utterance_clusters: npt.NDArray[np.int_]) -> "ProbablisticClustering":
        self.utterance_embeddings = utterance_embeddings
        self.utterance_clusters = utterance_clusters
        self.cluster_size = np.sqrt(
            ((cluster_centers[utterance_clusters] - utterance_embeddings)**2).sum(axis=-1).mean())
        return self

    def predict(self, utterance_embeddings: np.array) -> np.array:
        probabilities = np.zeros(
            (len(utterance_embeddings), self.n_clusters), dtype=int)
        distances = pairwise_distances(
            utterance_embeddings, self.utterance_embeddings) < self.cluster_size
        for j in range(len(self.utterance_embeddings)):
            probabilities[:, self.utterance_clusters[j]] += distances[:, j]
        return probabilities
