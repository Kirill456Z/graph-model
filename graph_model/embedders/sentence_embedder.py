import numpy as np
import typing as tp

from sentence_transformers import SentenceTransformer

from graph_model.embedders import OneViewEmbedder
from graph_model.dataset import DialogueDataset, Dialogue
import graph_model.embedders


class SentenceEmbedder(OneViewEmbedder):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', **config: tp.Any):
        super().__init__(config)
        self.config = config
        self.model = SentenceTransformer(model_name, **config)

    def encode_dialogue(self, dialogue: Dialogue) -> np.array:
        utterances = [utt.utterance for utt in dialogue]
        embeddings = self.model.encode(utterances)
        return embeddings

    def encode_dataset(self, dialogues: DialogueDataset) -> np.array:
        return self.model.encode(dialogues.utterances)

    def encode_new_dialogue(self, dialogue: Dialogue):
        return self.encode_dialogue(dialogue)

    def encode_new_dataset(self, dialogues: DialogueDataset):
        return self.encode_dataset(dialogues)


class SentenceEmbedderMP(OneViewEmbedder):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', **config: tp.Any):
        super().__init__(config)
        self.config = config
        self.model = SentenceTransformer(model_name, **config)

    def encode_dialogue(self, dialogue: Dialogue) -> np.array:
        utterances = [utt.utterance for utt in dialogue]
        embeddings = self.model.encode(utterances)
        return embeddings

    def encode_dataset(self, dialogues: DialogueDataset) -> np.array:
        pool = self.model.start_multi_process_pool()
        embeddings = self.model.encode_multi_process(dialogues.utterances, pool)
        self.model.stop_multi_process_pool(pool)
        return embeddings

    def encode_new_dialogue(self, dialogue: Dialogue):
        return self.encode_dialogue(dialogue)

    def encode_new_dataset(self, dialogues: DialogueDataset):
        return self.encode_dataset(dialogues)
