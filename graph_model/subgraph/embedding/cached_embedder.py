import numpy as np
from typing import List
from dataset import DialogueDataset, Dialogue

class CachedEmbedder:
    def __init__(self, dataset: DialogueDataset, embeddings: np.array):
        self.dataset = dataset
        self.embeddings = embeddings
    
    def encode_dialogue(self, dialogue: Dialogue) -> np.array:
        idx = self.dataset.get_dialog_start_idx(dialogue)
        return self.embeddings[idx:idx + len(dialogue)]
    
    def encode_dialogues(self, dialogues: List[Dialogue]) -> List[np.array]:
        return [self.encode_dialogue(dialogue) for dialogue in dialogues]

def merge_cached_embedders(x: CachedEmbedder, y: CachedEmbedder):
    return CachedEmbedder(x.dataset + y.dataset, np.concatenate((x.embeddings, y.embeddings), axis=0))
    
