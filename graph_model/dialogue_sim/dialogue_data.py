from dataclasses import dataclass
import numpy as np

@dataclass
class DialogueData:
    def _get_second_stage_emb(self, dial, data, embeddings, graph):
        if len(embeddings) != len(dial):
            start_idx = data.get_dialog_start_idx(dial)
        else:
            start_idx = 0
        self.first_stage_clusters = []
        self.first_stage_emb = []
        self.second_stage_clusters = []
        self.lm_embeddings = []
        for i in range(len(dial.utterances)):
            lm_embedd = embeddings[start_idx + i, :].ravel()
            self.lm_embeddings.append(lm_embedd)
            cluster = graph.one_stage_clustering.predict_cluster(lm_embedd, dial[i], dial).id
            self.first_stage_emb.append(graph.cluster_embeddings[cluster])
            self.first_stage_clusters.append(cluster)
            self.second_stage_clusters.append(graph.cluster_kmeans_labels[0][cluster])
        self.lm_embeddings = np.array(self.lm_embeddings)
        self.first_stage_emb = np.array(self.first_stage_emb)
    
    def __getitem__(self, key):
        return self.key_to_attr[key]
    
    def get_string(self):
        return "".join(self.utterances)

    def __init__(self, dial, data, embeddings, graph):
        utts = [str(utt) for utt in dial.utterances]
        self.utterances = utts
        self.services = dial.meta['services']
        self._get_second_stage_emb(dial, data,embeddings, graph)
        self.transitions = np.array(graph.get_transitions())
        self.key_to_attr = {
            'first_stage_emb' : self.first_stage_emb,
            'length' : len(self.first_stage_emb),
        }
    
    def get_utterances(self):
        res = []
        for i, utt in enumerate(self.utterances):
            res.append(Utterance(self.second_stage_clusters[i], self.first_stage_emb[i], utt))
        return res
    
class Utterance:
    def __init__(self, second_stage_cluster, first_stage_emb, string):
        self.second_stage_cluster = second_stage_cluster
        self.first_stage_emb = first_stage_emb
        self.string = string