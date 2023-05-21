from subgraph.tools import flatten
import numpy as np
import numpy.typing as npt
from typing import List, MutableSet


class FrequencyDialogueGraph:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.fitted = False

    def normalize(self):
        for i in range(self.n_clusters):
            self.frequencies[i] /= self.frequencies[i].sum()

    def fit(self, dialogues_clusters: List[npt.NDArray[np.int_]]) -> "FrequencyDialogueGraph":
        self.frequencies = np.zeros(
            (self.n_clusters, self.n_clusters)) + 1e-5
        for utterances_clusters in dialogues_clusters:
            for i in range(len(utterances_clusters) - 1):
                self.frequencies[utterances_clusters[i],
                                 utterances_clusters[i+1]] += 1
        self.normalize()
        self.fitted = True
        return self

    def fit_from_pretrained(self, frequencies: np.ndarray) -> "FrequencyDialogueGraph":
        self.frequencies = frequencies
        self.normalize()
        self.fitted = True
        return self


class RankedDialogueGraph:
    def __init__(self, fdd: FrequencyDialogueGraph):
        self.n_clusters = fdd.n_clusters
        self.ranking = np.zeros(
            (self.n_clusters, self.n_clusters),  dtype=np.int32)
        self.inverse_ranking = np.zeros(
            (self.n_clusters, self.n_clusters),  dtype=np.int32)
        for i in range(self.n_clusters):
            self.ranking[i] = np.flip(np.argsort(fdd.frequencies[i]))
            self.inverse_ranking[i] = np.argsort(self.ranking[i])

    def rank_dialogue(self, utterances_clusters: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        return np.array(list(self.inverse_ranking[utterances_clusters[i]][utterances_clusters[i+1]] for i in range(len(utterances_clusters) - 1)))

    def rank_dialogues(self, dialogues_clusters: List[npt.NDArray[np.int_]]) -> List[npt.NDArray[np.int_]]:
        return [self.rank_dialogue(dialogue) for dialogue in dialogues_clusters]


class ClusterEmbedder:
    def __init__(self, embeddings):
        self.frequencies = embeddings
        self._fit = True

    def encode_utterance(self, utterance_cluster: int) -> np.array:
        assert self._fit
        return self.frequencies[utterance_cluster]

    def encode_dialogue(self, dialogue_clustering: List[int]) -> np.array:
        return [self.encode_utterance(utterance_c) for utterance_c in dialogue_clustering]

    def encode_dialogues(self, dialogues_clustering: List[npt.NDArray[np.int_]]) -> List[np.array]:
        return [self.encode_dialogue(dialogue) for dialogue in dialogues_clustering]


class ContextDialogueGraph:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self._fit = False

    def normalize(self):
        for i in range(self.n_clusters):
            self.frequencies[i] /= self.frequencies[i].sum()

    def fit(self, dialogues_clusters: List[npt.NDArray[np.int_]], window: int = None) -> "ContextDialogueGraph":
        self.frequencies = np.zeros(
            (self.n_clusters, self.n_clusters)) + 1e-5
        if window == None:
            for utterances_clusters in dialogues_clusters:
                for i in utterances_clusters:
                    for j in utterances_clusters:
                        self.frequencies[i][j] += 1
        else:
            for utterances_clusters in dialogues_clusters:
                for i in range(len(utterances_clusters)):
                    for j in range(max(0, i - window),
                                   min(i + window + 1, len(utterances_clusters))):
                        self.frequencies[utterances_clusters[i]
                                         ][utterances_clusters[j]] += 1
        self.normalize()
        self._fit = True
        return self

    def encode_utterance(self, utterance_cluster: int) -> np.array:
        assert self._fit
        return self.frequencies[utterance_cluster]

    def encode_dialogue(self, dialogue_clustering: List[int]) -> np.array:
        return [self.encode_utterance(utterance_c) for utterance_c in dialogue_clustering]

    def encode_dialogues(self, dialogues_clustering: List[npt.NDArray[int]]) -> List[np.array]:
        return [self.encode_dialogue(dialogue) for dialogue in dialogues_clustering]


class CenteredDialogueGraph:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.fitted = False

    def fit(self, dialogues_clusters: List[npt.NDArray[np.int_]], dialogue_embeddings: List[np.array]) -> "CenteredDialogueGraph":
        # Initializing arrays
        self.frequencies = np.zeros(
            (self.n_clusters, self.n_clusters), dtype=np.int32)

        embedding_dim = len(dialogue_embeddings[0][0])
        self.centers = np.zeros(
            (self.n_clusters, self.n_clusters, embedding_dim))
        self.ranking = np.zeros_like(self.frequencies)

        # Caclulating moves
        for utterances_clusters, utrrerances_embeddings in zip(dialogues_clusters, dialogue_embeddings):
            assert len(utterances_clusters) == len(utrrerances_embeddings)
            for i in range(len(utterances_clusters) - 1):
                self.frequencies[utterances_clusters[i],
                                 utterances_clusters[i+1]] += 1
                self.centers[utterances_clusters[i],
                             utterances_clusters[i+1]] += utrrerances_embeddings[i+1]
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                if self.frequencies[i][j] == 0:
                    continue
                self.centers[i][j] = self.centers[i][j] / \
                    self.frequencies[i][j]

        # Ranking
        self.ranking = np.zeros(
            (self.n_clusters, self.n_clusters),  dtype=np.int32)
        self.inverse_ranking = np.zeros(
            (self.n_clusters, self.n_clusters),  dtype=np.int32)
        for i in range(self.n_clusters):
            self.ranking[i] = np.flip(np.argsort(self.frequencies[i]))
            self.inverse_ranking[i] = np.argsort(self.ranking[i])

        # Done
        self.fitted = True
        return self

    def predict(self, utterance_clusters: npt.NDArray[np.int_]) -> np.array:
        next_clusters = self.predict_next_clusters(utterance_clusters)
        return self.centers[utterance_clusters, next_clusters]

    def predict_next_clusters(self, utterance_clusters: npt.NDArray[np.int_]) -> np.array:
        next_clusters = self.ranking[utterance_clusters][:, 0]
        return next_clusters

    def predict_next_clusters_probablistic(self, uc_probabilities: npt.NDArray[np.int32]):
        cluster_probs = np.matmul(uc_probabilities, self.frequencies)
        best_cluster = np.zeros(len(uc_probabilities), dtype=np.int32)
        for i in range(len(cluster_probs)):
            best_cluster[i] = np.argsort(cluster_probs[i])[-1]
        return best_cluster

    def predict_probablistic(self, uc_probabilities: npt.NDArray[np.int32]):
        next_clusters = self.predict_next_clusters_probablistic(
            uc_probabilities)
        preds = np.zeros((len(uc_probabilities), len(self.centers[0, 0])))
        for i in range(len(uc_probabilities)):
            preds[i] = np.sum(self.centers[:, next_clusters[i]] *
                              uc_probabilities[i][:, None], axis=0) / uc_probabilities[i].sum()
        return preds


class CondensedCentredDialogueGraph:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.fit_ = False

    def fit(self, dialogues_clusters: List[npt.NDArray[np.int_]], dialogue_embeddings: List[np.array]) -> "CondensedCentredDialogueGraph":

        # Initializing arrays
        self.frequencies = np.zeros(
            (self.n_clusters, self.n_clusters), dtype=np.int32)
        self.next_cluster = np.zeros_like(self.frequencies)

        # Caclulating moves
        for utterances_clusters in dialogues_clusters:
            for i in range(len(utterances_clusters) - 1):
                self.frequencies[utterances_clusters[i],
                                 utterances_clusters[i+1]] += 1

        # Ranking
        self.next_cluster = np.zeros(
            (self.n_clusters, self.n_clusters),  dtype=np.int32)
        self.inverse_ranking = np.zeros(
            (self.n_clusters, self.n_clusters),  dtype=np.int32)
        for i in range(self.n_clusters):
            self.next_cluster[i] = np.flip(np.argsort(self.frequencies[i]))
            self.inverse_ranking[i] = np.argsort(self.next_cluster[i])

        # Calculating best centers
        embedding_dim = len(dialogue_embeddings[0][0])
        self.centers = np.zeros((self.n_clusters, embedding_dim))
        for utterances_clusters, utterance_embeddings in zip(dialogues_clusters, dialogue_embeddings):
            assert len(utterances_clusters) == len(utterance_embeddings)
            for i in range(len(utterances_clusters) - 1):
                if self.next_cluster[utterances_clusters[i]][0] == utterances_clusters[i+1]:
                    self.centers[utterances_clusters[i]] += utterance_embeddings[i+1] / \
                        self.frequencies[utterances_clusters[i],
                                         utterances_clusters[i+1]]

        # Done
        self.fit_ = True
        return self

    def predict(self, utterance_clusters: npt.NDArray[np.int_]) -> np.array:
        return self.centers[utterance_clusters]

    def predict_next_clusters(self, utterance_clusters: npt.NDArray[np.int_]) -> np.array:
        next_clusters = self.next_cluster[utterance_clusters][:, 0]
        return next_clusters


def accuracy_at_k(ranked_dialogues: List[npt.NDArray[np.int_]], at_k=[1, 3, 5]) -> List[int]:
    ranks = flatten(ranked_dialogues)
    return [(ranks < k).mean() for k in at_k]


def induced_subgraph(fdd: FrequencyDialogueGraph, vertices: MutableSet) -> FrequencyDialogueGraph:
    frequencies = np.copy(fdd.frequencies)
    for i in range(fdd.n_clusters):
        for j in range(fdd.n_clusters):
            if j not in vertices:
                frequencies[i][j] = 1e-5
    return FrequencyDialogueGraph(n_clusters=fdd.n_clusters).fit_from_pretrained(frequencies)
