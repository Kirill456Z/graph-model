from sentence_transformers import SentenceTransformer
from collections import defaultdict
from graph_model import dialogue_graph 
import pickle
import numpy as np
import sys
import torch
from graph_model.cluster_selection.cluster_selection import PredictNextCluster, prepare_input

from graph_model import dialogue_graph
from graph_model import dataset
from graph_model import clustering
from graph_model import embedders
from graph_model import dialogue_sim

sys.modules["dialogue_graph"] = dialogue_graph
sys.modules["dataset"] = dataset
sys.modules["clustering"] = clustering
sys.modules["embedders"] = embedders
sys.modules["dialogue_sim"] = dialogue_sim

class Utterance:
    def __init__(self, string, embedding):
        self.string = string
        self.embedding = embedding

class GraphModel:
    def __init__(self):
        self.cluster_prediction = PredictNextCluster(403, 512, 11)
        self.cluster_prediction.load_state_dict(torch.load("graph_model/cluster_selection/cluster_selection.pt", map_location=torch.device('cpu')))
        self.cluster_prediction.eval()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        with open("graph_model/dialogue_sim/graphs/one_serv_10_taxi.pkl", 'rb') as file:
            self.graph = pickle.load(file)
        with open("graph_model/dialogue_sim/data/train_dials.pkl", "rb") as file:
            self.train_dials = pickle.load(file)
        self.transitions = self.train_dials[0].transitions

        self.cluster_to_utts = defaultdict(list)
        for dial in self.train_dials:
            for i in range(len(dial.utterances)):
                self.cluster_to_utts[dial.second_stage_clusters[i]].append(Utterance(dial.utterances[i], dial.lm_embeddings[i])) 

        self.cluster_to_embs = {}
        for i, utts in self.cluster_to_utts.items():
            embs = []
            for utt in utts:
                embs.append(utt.embedding)
            self.cluster_to_embs[i] = np.array(embs)

    def get_closest(self, target, utt_embs, k = 5):
        similarity = (utt_embs @ target.reshape(-1, 1)).ravel()
        closest = np.argsort(similarity)[:-k:-1]
        return closest

    def get_next_cluster(self, cur_cluster):
        return np.argmax(self.transitions[cur_cluster])
    
    def __call__(self, text):
        embedding = self.model.encode(text)
        one_stage_cluster = self.graph.one_stage_clustering._subclusters["USER"].predict_cluster(embedding[0]).id
        second_stage_cluster = self.graph.cluster_kmeans_labels[0][one_stage_cluster]

        prepared_input = prepare_input(embedding[0], self.transitions[second_stage_cluster], self.graph.cluster_embeddings[one_stage_cluster] )
        #next_cluster = self.get_next_cluster(second_stage_cluster)
        next_cluster = torch.argmax(self.cluster_prediction(prepared_input.reshape(1, -1)), 1).item()
        closest_idxs = self.get_closest(embedding, self.cluster_to_embs[next_cluster])
        closest_str = []
        for i in closest_idxs:
            closest_str.append(self.cluster_to_utts[next_cluster][i].string)
        return [closest_str[0]], next_cluster