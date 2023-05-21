from dataset.focus import load_focus
from minicache import cached_step, cache_for
from dataset import DialogueDataset

### Common
from subgraph.embedding.cached_embedder import CachedEmbedder, merge_cached_embedders
from subgraph.tools import flatten
import typing as tp
import numpy as np


@cached_step()
def head(data):
    if isinstance(data, DialogueDataset):
        return DialogueDataset(data[:5])
    return data[:5]

@cached_step()
def embedded_up_dataset(dataset: DialogueDataset, embedder: CachedEmbedder) -> tp.Tuple[np.array, np.array]:
    dialogue_embeddings = embedder.encode_dialogues(dataset)
    x, y = [], []
    for utterance_embeddings in dialogue_embeddings:
        x.append(utterance_embeddings[:-1])
        y.append(utterance_embeddings[1:])
    return flatten(x), flatten(y)

@cached_step()
def convert_embeddings(dataset: tp.Tuple[DialogueDataset, DialogueDataset]) -> tp.Tuple[np.array, np.array]:
    assert False, "Could not load convert embeddings for that dataset, is overload imported?"


@cached_step()
def convert_embedder(dataset: tp.Tuple[DialogueDataset, DialogueDataset]) -> CachedEmbedder:
    train, test = dataset
    train_emb, test_emb = convert_embeddings(dataset)
    embedder = merge_cached_embedders(CachedEmbedder(
        test, test_emb), CachedEmbedder(train, train_emb))
    return embedder


@cached_step()
def dataset_head(dataset: tp.Tuple):
    return head(train(dataset)), head(test(dataset))

@cached_step()
def train(data):
    return data[0]


@cached_step()
def test(data):
    return data[1]


@cached_step()
def features(data):
    return data[0]


@cached_step()
def targets(data):
    return data[1]


### MULTIWOZ
from dataset.multiwoz import load_multiwoz
from dataset import DialogueDataset
from pathlib import Path
import os


@cached_step()
def multiwoz():
    SRC_DIR = Path(os.getcwd())
    MULTIWOZ_PATH = SRC_DIR / "multiwoz/data/MultiWOZ_2.2"
    EMB_PATH = SRC_DIR / "embeddings/multiwoz"

    test = DialogueDataset.from_miltiwoz_v22(
        load_multiwoz('test', MULTIWOZ_PATH,
        order = ['dialogues_001.json', 'dialogues_002.json']))
    train = DialogueDataset.from_miltiwoz_v22(
        load_multiwoz('train', MULTIWOZ_PATH,
                      order=[
                          'dialogues_001.json', 'dialogues_011.json', 'dialogues_007.json', 'dialogues_010.json',
                          'dialogues_017.json', 'dialogues_005.json', 'dialogues_015.json', 'dialogues_012.json',
                          'dialogues_016.json', 'dialogues_013.json', 'dialogues_004.json', 'dialogues_009.json',
                          'dialogues_003.json', 'dialogues_006.json', 'dialogues_008.json', 'dialogues_002.json',
                          'dialogues_014.json'
                      ])
    )

    return train, test


@cache_for("convert_embeddings", "multiwoz")
def convert_embeddings_mutliwoz():
    SRC_DIR = Path(os.getcwd())
    EMB_PATH = SRC_DIR / "embeddings/embeddings/multiwoz"
    with open(EMB_PATH / "emb_train.pkl", "rb") as f:
        train_emb = np.load(f, allow_pickle=True)

    with open(EMB_PATH / "emb_test.pkl", "rb") as f:
        test_emb = np.load(f, allow_pickle=True)
    return train_emb, test_emb


# AMAZONQA
from dataset.amazonqa import load_amazonqa
from dataset import DialogueDataset
from pathlib import Path
import os


@cached_step()
def amazonqa():
    SRC_DIR = Path(os.getcwd())
    AMAZONQA_PATH = SRC_DIR / "amazonqa/processed"

    test = DialogueDataset.from_qa(load_amazonqa(AMAZONQA_PATH, "test"))
    train = DialogueDataset.from_qa(load_amazonqa(AMAZONQA_PATH, "train"))

    return train, test


@cache_for("convert_embeddings", "amazonqa")
def convert_embeddings_amazonqa():
    SRC_DIR = Path(os.getcwd())
    EMB_PATH = SRC_DIR / "embeddings/embeddings/amazonqa"
    with open(EMB_PATH / "emb_train.pkl", "rb") as f:
        train_emb = np.load(f, allow_pickle=True)

    with open(EMB_PATH / "emb_test.pkl", "rb") as f:
        test_emb = np.load(f, allow_pickle=True)
    return train_emb, test_emb

### PERSONACHAT

from dataset.personachat import load_personachat
from dataset import DialogueDataset
from pathlib import Path
import os


@cached_step()
def personachat():
    SRC_DIR = Path(os.getcwd())
    PERSONACHAT_PATH = SRC_DIR / "personachat/processed"

    test = DialogueDataset.from_custom_dataset(
        load_personachat(PERSONACHAT_PATH, "test"))
    train = DialogueDataset.from_custom_dataset(
        load_personachat(PERSONACHAT_PATH, "train"))

    return train, test


@cache_for("convert_embeddings", "personachat")
def convert_embeddings_personachat():
    SRC_DIR = Path(os.getcwd())
    EMB_PATH = SRC_DIR / "embeddings/embeddings/personachat"
    with open(EMB_PATH / "emb_train.pkl", "rb") as f:
        train_emb = np.load(f, allow_pickle=True)

    with open(EMB_PATH / "emb_test.pkl", "rb") as f:
        test_emb = np.load(f, allow_pickle=True)
    return train_emb, test_emb


### DailyDialog
from datasets import load_dataset
from dataset import DialogueDataset
from pathlib import Path
import os
import numpy as np


@cached_step()
def dailydialog():
    data = load_dataset("daily_dialog")
    test = DialogueDataset.from_dailydialog(data['test'])
    train = DialogueDataset.from_dailydialog(data['train'])

    return train, test

@cache_for("convert_embeddings", "dailydialog")
def convert_embeddings_dailydialog():
    SRC_DIR = Path(os.getcwd())
    EMB_PATH = SRC_DIR / "embeddings/embeddings/daily_dialog"
    with open(EMB_PATH / "emb_train.pkl", "rb") as f:
        train_emb = np.load(f, allow_pickle=True)

    with open(EMB_PATH / "emb_test.pkl", "rb") as f:
        test_emb = np.load(f, allow_pickle=True)
    return train_emb, test_emb

### Open subtitles
from dataset import DialogueDataset
from dataset.opensubtitles import load_opensubtitles
from pathlib import Path
import os


@cached_step()
def opensubtitles():
    SRC_DIR = Path(os.getcwd())
    OPENSUBTITLES_PATH = SRC_DIR / "opensubtitles"

    train = DialogueDataset.from_custom_dataset(
        load_opensubtitles(OPENSUBTITLES_PATH, "train"))
    test = DialogueDataset.from_custom_dataset(
        load_opensubtitles(OPENSUBTITLES_PATH, "test"))

    return train, test

@cache_for("convert_embeddings", "opensubtitles")
def convert_embeddings_opensubtitles():
    SRC_DIR = Path(os.getcwd())
    EMB_PATH = SRC_DIR / "embeddings/embeddings/open_subtitles"
    with open(EMB_PATH / "emb_train.pkl", "rb") as f:
        train_emb = np.load(f, allow_pickle=True)

    with open(EMB_PATH / "emb_test.pkl", "rb") as f:
        test_emb = np.load(f, allow_pickle=True)
    return train_emb, test_emb

### FOCus

from dataset import DialogueDataset
from dataset.focus import load_focus
from pathlib import Path
import os


@cached_step()
def focus():
    SRC_DIR = Path(os.getcwd())
    FOCUS_PATH = SRC_DIR / "focus/processed"

    train = DialogueDataset.from_custom_dataset(
        load_focus(FOCUS_PATH, "train"))
    test = DialogueDataset.from_custom_dataset(
        load_focus(FOCUS_PATH, "val"))

    return train, test


@cache_for("convert_embeddings", "focus")
def convert_embeddings_focus():
    SRC_DIR = Path(os.getcwd())
    EMB_PATH = SRC_DIR / "embeddings/embeddings/focus"
    with open(EMB_PATH / "emb_train.pkl", "rb") as f:
        train_emb = np.load(f, allow_pickle=True)

    with open(EMB_PATH / "emb_val.pkl", "rb") as f:
        test_emb = np.load(f, allow_pickle=True)
    return train_emb, test_emb
