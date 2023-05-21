import typing as tp

from graph_model.dataset import Dialogue, Utterance


def speaker_filter(utterance: Utterance, dialogue: Dialogue) -> tp.Hashable:
    return utterance.speaker


def default_filter(utterance: Utterance, dialogue: Dialogue) -> tp.Hashable:
    return "data"
