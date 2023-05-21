from minicache import pickle_load, pickle_store, cached_step
from subgraph.model import Sequential
from subgraph.datasets import features, targets

import numpy as np
import typing as tp

@cached_step(load=pickle_load, store=pickle_store)
def recall_natk(n: int, k: int, data: tp.Tuple[np.array, np.array], model: Sequential, seed: int):
    generator = np.random.default_rng(seed)
    preds = model.forward_step(features(data))
    golds = model.forward(targets(data))
    score = 0.0
    for i, pred in enumerate(preds):
        # Perorming set ...
        sample_set = generator.choice(
            golds[:i] if i > len(preds) / 2 else golds[i+1:], n)
        gold_index = generator.choice(n, 1).item()
        sample_set[gold_index] = golds[i]

        # Ranking
        distances = ((sample_set - pred)**2).sum(axis=-1)
        ranks = np.argsort(np.argsort(distances))
        if ranks[gold_index] < k:
            score += 1
    return score / len(preds)
