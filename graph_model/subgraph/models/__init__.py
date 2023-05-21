from minicache import cached_step
from subgraph.embedding.cached_embedder import CachedEmbedder
from subgraph.model import Model, Sequential, Step, StepMap
from subgraph.graph import FrequencyDialogueGraph, RankedDialogueGraph, CenteredDialogueGraph, CondensedCentredDialogueGraph
from subgraph.clustering.kmeans import kmeans 
from subgraph.datasets import train
from subgraph.models.probablistic import ProbablisticClustering

from dataset import DialogueDataset
import typing as tp
import numpy as np



@cached_step()
def model_fdg(dataset: tp.Tuple[DialogueDataset, DialogueDataset], embedder: CachedEmbedder, n_clusters: int, seed: int):
    clustering = kmeans(dataset, embedder, n_clusters, seed)
    rdd = RankedDialogueGraph(FrequencyDialogueGraph(n_clusters).fit(clustering.predict_dialogues(embedder.encode_dialogues(train(dataset)))))
    return Sequential([
        Step(clustering.predict_utterances),
        StepMap(lambda y: rdd.ranking[y][0]),
        StepMap(lambda y: clustering.clustering.cluster_centers_[y])
    ])

@cached_step()
def model_centered_fdg(dataset: tp.Tuple[DialogueDataset, DialogueDataset], embedder: CachedEmbedder, n_clusters: int, seed :int):
    clustering = kmeans(dataset, embedder, n_clusters, seed)
    train_emb = embedder.encode_dialogues(train(dataset))
    centered_fdg = CenteredDialogueGraph(n_clusters).fit(clustering.predict_dialogues(train_emb), train_emb)

    return Sequential([
        Step(clustering.predict_utterances),
        Step(centered_fdg.predict)
    ])

@cached_step()
def model_cocentered_fdg(dataset: tp.Tuple[DialogueDataset, DialogueDataset], embedder: CachedEmbedder, n_clusters: int, seed :int):
    clustering = kmeans(dataset, embedder, n_clusters, seed)
    train_emb = embedder.encode_dialogues(train(dataset))
    centered_fdg = CondensedCentredDialogueGraph(n_clusters).fit(clustering.predict_dialogues(train_emb), train_emb)

    return Sequential([
        Step(clustering.predict_utterances),
        Step(centered_fdg.predict)
    ])

@cached_step()
def model_centered_fdg_probablistic(dataset: tp.Tuple[DialogueDataset, DialogueDataset], train_emb: np.array, embedder: CachedEmbedder, n_clusters: int, seed: int):
    clustering = kmeans(dataset, embedder, n_clusters, seed)
    train_embed = embedder.encode_dialogues(train(dataset))
    pc = ProbablisticClustering(n_clusters).fit(clustering.clustering.cluster_centers_, train_emb, clustering.predict_utterances(train_emb))
    centered_fdg = CenteredDialogueGraph(n_clusters).fit(clustering.predict_dialogues(train_embed), train_embed)

    return Sequential([
        Step(pc.predict),
        Step(centered_fdg.predict_probablistic)
    ])

from subgraph.edgecluster.tools import next_utterance_emb, closest_in_train_embedder, nearest_utterances_raw
from subgraph.tools import flatten
from subgraph.datasets import train, test

@cached_step()
def model_one_neighbor(dataset: tp.Tuple[DialogueDataset, DialogueDataset], embedder: CachedEmbedder):
    new_train_emb = next_utterance_emb(train(dataset), embedder)
    train_emb = flatten(embedder.encode_dialogues(train(dataset)))
    return Sequential([
        Step(lambda x: new_train_emb[nearest_utterances_raw(train_emb, x)])
    ])