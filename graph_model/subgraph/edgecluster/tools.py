import numpy as np
from typing import Tuple
from sklearn.neighbors import NearestNeighbors

from dataset import DialogueDataset, Dialogue, Utterance
from minicache import cached_step

from subgraph.embedding.cached_embedder import CachedEmbedder, merge_cached_embedders
from subgraph.tools import flatten
from subgraph.datasets import train, test

def nearest_utterances_raw(train_emb: np.array, test_emb: np.array) -> np.array:
    neigbours = NearestNeighbors(n_neighbors=1).fit(train_emb)
    _, result = neigbours.kneighbors(test_emb)
    return flatten(result)


@cached_step(ignore_cache=True)
def nearest_utterances(dataset: Tuple[DialogueDataset, DialogueDataset], embedder: CachedEmbedder) -> np.array:
    train, test = dataset
    train_emb = flatten(embedder.encode_dialogues(train))
    test_emb = flatten(embedder.encode_dialogues(test))
    return nearest_utterances_raw(train_emb, test_emb)


@cached_step(ignore_cache=True)
def closest_in_train_embedder(new_train_emb: np.array, dataset: Tuple[DialogueDataset, DialogueDataset], old_embedder: CachedEmbedder) -> CachedEmbedder:
    new_test_emb = new_train_emb[nearest_utterances(dataset, old_embedder)]
    return merge_cached_embedders(CachedEmbedder(train(dataset), new_train_emb), CachedEmbedder(test(dataset), new_test_emb))


@cached_step(ignore_cache=True)
def next_utterance_emb(train: DialogueDataset, embedder: CachedEmbedder):
    result = []
    for dialogue in embedder.encode_dialogues(train):
        result.append(dialogue[1:])
        result.append(dialogue[0:1])
    
    return np.concatenate(result)
