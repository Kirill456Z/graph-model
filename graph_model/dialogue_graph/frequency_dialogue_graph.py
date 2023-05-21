from collections import defaultdict
import numpy as np
import scipy.sparse

from copy import deepcopy
import typing as tp
import pickle
import os

from graph_model.clustering.filters import default_filter
from graph_model.clustering.subclustering import SubClustering

from graph_model.dataset import DialogueDataset, Dialogue
from graph_model.embedders.interface import OneViewEmbedder
from graph_model.clustering.interface import OneViewClustering


class FrequencyDialogueGraph:
    def __init__(
        self,
        dialogues: DialogueDataset,
        embedder: OneViewEmbedder,
        clustering: OneViewClustering,
    ):
        self.dialogues: DialogueDataset = dialogues
        self.clustering: OneViewClustering = clustering
        self.embedder: OneViewEmbedder = embedder

        self.n_vertices = clustering.get_nclusters() + 1
        self.start_node = self.n_vertices - 1

        self.edges = [[0] * self.n_vertices for _ in range(self.n_vertices)]

        self.eps = 1e-5

    #         self._build()

    def _add_dialogue(self, dialogue: Dialogue) -> None:
        utt_idx = self.dialogues.get_dialog_start_idx(dialogue)
        current_node = self.start_node
        for _ in dialogue:
            next_node = self.clustering.get_utterance_cluster(utt_idx).id
            self.edges[current_node][next_node] += 1
            current_node = next_node
            utt_idx += 1

    def build(self):
        for dialogue in self.dialogues:
            self._add_dialogue(dialogue)

        self.probabilities = []
        for v in range(self.n_vertices):
            sum = np.sum(self.edges[v])
            if sum > 0:
                self.probabilities.append(np.array(self.edges[v]) / sum)
            else:
                self.probabilities.append(np.array(self.edges[v]).astype(np.float32))

    def iter_dialogue(self, dialogue: Dialogue):
        d_embs = self.embedder.encode_new_dialogue(dialogue)
        if isinstance(d_embs, scipy.sparse.spmatrix):
            d_embs = d_embs.toarray()
        for utt, emb in zip(dialogue, d_embs):
            next_node = self.clustering.predict_cluster(emb, utt, dialogue).id
            yield next_node, emb
    
    def get_edges(self) -> np.array:
        return np.array(deepcopy(self.edges), dtype=np.int32)

    def get_dataset_markup(self, dataset: DialogueDataset):
        labels = []
        for dialogue in dataset:
            for next_node, _ in self.iter_dialogue(dialogue):
                labels.append(next_node)
        return labels

    def get_transitions(self):
        return deepcopy(self.probabilities)

    def _dialogue_success_rate(
        self, dialogue: Dialogue, cluster_frequencies, separator, acc_ks=None
    ):
        if acc_ks is None:
            acc_ks = []
        acc_ks = np.array(acc_ks)

        d_embs = self.embedder.encode_new_dialogue(dialogue)
        if isinstance(d_embs, scipy.sparse.spmatrix):
            d_embs = d_embs.toarray()

        logprob = 0
        accuracies = {}
        total_utt = {}

        visited_clusters = []

        current_node = self.start_node
        for utt, emb in zip(dialogue, d_embs):
            filter_val = separator(utt, dialogue)
            if filter_val not in accuracies.keys():
                accuracies[filter_val] = np.zeros(len(acc_ks))
                total_utt[filter_val] = 0
            total_utt[filter_val] += 1
            next_node = self.clustering.predict_cluster(emb, utt, dialogue).id
            cluster_frequencies[next_node] += 1
            visited_clusters.append(next_node)
            prob = self.probabilities[current_node][next_node]
            prob = max(prob, self.eps)
            logprob -= np.log(prob) * prob

            next_cluster_ind = (self.probabilities[current_node] >= prob).sum()
            accuracies[filter_val] = accuracies[filter_val] + (
                next_cluster_ind <= acc_ks
            )

            current_node = next_node
        for key in accuracies.keys():
            accuracies[key] /= total_utt[key]
        unique_score = len(np.unique(visited_clusters)) / len(visited_clusters)
        return logprob, unique_score, accuracies

    def success_rate(self, test: DialogueDataset, acc_ks=None, separator=None):
        if acc_ks is None:
            acc_ks = []

        if separator is None:
            separator = default_filter

        logprob = 0
        accuracies = {}
        cluster_frequencies = np.zeros(self.n_vertices - 1)
        unique_score = 0.0
        for dialogue in test:
            lp, us, acc = self._dialogue_success_rate(
                dialogue, cluster_frequencies, acc_ks=acc_ks, separator=separator
            )
            logprob += lp
            unique_score += us

            if len(accuracies) == 0:
                accuracies = acc
            else:
                for key in accuracies.keys():
                    accuracies[key] += acc[key]
        logprob /= len(test)
        for key in accuracies.keys():
            accuracies[key] /= len(test)
        unique_score /= len(test)
        return logprob, cluster_frequencies, unique_score, accuracies

    def get_node_content(self, idx: int) -> np.array:
        utt_indices = self.clustering.get_cluster(idx).utterances
        return np.array(self.dialogues.utterances)[utt_indices]

    def data_by_clusters(
        self, data: DialogueDataset, data_emb: np.array
    ) -> tp.Dict[int, list]:
        divided = defaultdict(list)
        utt_idx = 0
        for dialogue in data:
            for utt in dialogue:
                cluster = self.clustering.predict_cluster(
                    data_emb[utt_idx], utt, dialogue
                )
                divided[cluster.id].append(utt_idx)
                utt_idx += 1
        return divided

    def save_state(
        self,
        filename: str,
        is_two_stage: bool,
        save_embedder: bool = False,
        save_dialogues: bool = False,
    ) -> tp.Dict[str, dict]:
        """
        Saves the graph state without the train dataset and the embedder.
        In order to lad the graph, the dataset must be provided separately.
        """
        clustering_state = vars(self.clustering)
        clustering_keys = set(clustering_state.keys()) - {"dialogues"}
        if is_two_stage:
            clustering_keys = clustering_keys - {"_subclustering"}

        graph_state = vars(self)
        graph_keys = set(graph_state.keys()) - {"clustering"}
        state_dict = {
            "subclustering": None,
            "subclustering_type": None,
            "clustering": {
                key: deepcopy(clustering_state[key]) for key in clustering_keys
            },
            "graph_state": {key: deepcopy(graph_state[key]) for key in graph_keys},
            "clustering_type": self.clustering.__class__,
        }

        if is_two_stage:
            state_dict["subclustering"] = self.clustering.get_subclustering_state()
            state_dict["subclustering_type"] = self.clustering.get_subclustering_type()

        if not save_embedder:
            del state_dict["graph_state"]["embedder"]

        if not save_dialogues:
            del state_dict["graph_state"]["dialogues"]

        dirname = os.path.dirname(filename)

        if not os.path.exists(dirname):
            print(f"Creating directory {dirname} ...")
            os.makedirs(dirname)

        with open(filename, "wb") as f:
            print(f"Saving the graph state to {filename} ...")
            pickle.dump(state_dict, f)

        return state_dict

    @classmethod
    def load_state(
        cls,
        filename: str,
        dialog_set: DialogueDataset = None,
        embedder: OneViewEmbedder = None,
    ) -> "FrequencyDialogueGraph":
        state_dict: tp.Dict[str, dict] = None
        with open(filename, "rb") as f:
            state_dict = pickle.load(f)

        if dialog_set is not None:
            state_dict["graph_state"]["dialogues"] = dialog_set

        state_dict["clustering"]["dialogues"] = state_dict["graph_state"]["dialogues"]

        if state_dict["subclustering"] is not None:
            state_dict["subclustering"]["dialogues"] = state_dict["graph_state"][
                "dialogues"
            ]
            state_dict["clustering"]["_subclustering"] = state_dict[
                "subclustering_type"
            ].from_dict(state_dict["subclustering"])

        if embedder is not None:
            state_dict["graph_state"]["embedder"] = embedder

        state_dict["graph_state"]["clustering"] = state_dict[
            "clustering_type"
        ].from_dict(state_dict["clustering"])
        return cls.from_dict(state_dict["graph_state"])

    @classmethod
    def from_dict(cls, fields: tp.Dict[str, tp.Any]) -> "SubClustering":
        object = cls.__new__(cls)
        for key in fields:
            setattr(object, key, fields[key])
        return object
