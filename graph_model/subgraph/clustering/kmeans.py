import numpy as np
import numpy.typing as npt
from typing import List
import sklearn.cluster


class KMeansClustering:
    def __init__(self, n_clusters=15, random_state=42, **config):
        self.clustering = sklearn.cluster.KMeans(n_clusters = n_clusters, random_state = random_state, **config)
        self.fitted = False

    def fit(self, utterance_embeddings: np.array) -> "KMeansClustering":
        self.clustering.fit(utterance_embeddings)
        self.fitted = True
        return self

    def predict_utterances(self, utterance_embeddings: np.array) -> npt.NDArray[np.int_]:
        assert self.fitted
        return self.clustering.predict(utterance_embeddings)

    def predict_dialogues(self, diglogue_embeddings: List[np.array]) -> List[npt.NDArray[np.int_]]:
        assert self.fitted
        return [self.predict_utterances(utterance_embeddings) for utterance_embeddings in diglogue_embeddings]


## STEP
from minicache import cached_step, pickle_load, pickle_store
from subgraph.tools import flatten
from subgraph.embedding.cached_embedder import CachedEmbedder
from dataset import DialogueDataset
import typing as tp
@cached_step(load=pickle_load, store=pickle_store)
def kmeans(dataset: tp.Tuple[DialogueDataset, DialogueDataset], embedder: CachedEmbedder, n_clusters: int, seed: int) -> KMeansClustering:
    train, test = dataset
    clustering = KMeansClustering(n_clusters, seed).fit(
        flatten(embedder.encode_dialogues(train)))
    return clustering

