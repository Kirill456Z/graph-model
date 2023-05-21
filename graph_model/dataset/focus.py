# !gdown https://drive.google.com/uc?id=1YmEW12HqjAjlEfZ05g8VLRux8kyUjdcI&export=download
# !unzip FoCus.zip

import json
from pathlib import Path
import os
import typing as tp

from tqdm import tqdm
import re


def validate_filename(path: str, split: str):
    if split not in ["train", "val"]:
        raise ValueError(
            'Invalid split type. Appropriate splits: ["train", "val"]'
        )

    split_prefix = "valid" if split == "val" else split
    extension = ".json"

    filename = "".join([split_prefix, "_focus", extension])

    data_dir = Path(path) / filename

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Given file: {data_dir} doesn't exist.")

    return filename, data_dir

def build_turn(id: int, speaker: str, utterance: str):
    return {
        "turn_id": id,
        "speaker": speaker,
        "utterance": utterance,
    }

def process_dialogue(dialogue_id: int, dialogue: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    result = {"dialogue_id": dialogue_id, "turns": []}
    utt_len = len(dialogue["utterance"])
    utterances = dialogue["utterance"][-1][f"dialogue{utt_len}"]
    speakers = ["HUMAN", "AGENT"]
    for i, utt in enumerate(utterances):
        result["turns"].append(build_turn(i, speakers[i % 2], utt))
    return result


def preprocess_data(path: str, split: str):
    filename, data_dir = validate_filename(path, split)
    new_data_dir = data_dir.parent / "processed" / Path(filename).with_suffix(".json")
    print(f"Preprocessing {data_dir}")
    new_data = []
    with data_dir.open() as f:
        data = json.load(f)["data"]
        for i, item in tqdm(enumerate(data)):
            new_data.append(process_dialogue(i, item))

    with new_data_dir.open("w+") as f:
        print(f"Writing to {new_data_dir}")
        f.write(json.dumps(new_data, indent=2))

def load_focus(path: str, split: str) -> tp.List[dict]:
    """
    Load preprocessed FoCus Dataset
    path - path to the dataset
    split - "val" / "train"
    """
    _, data_dir = validate_filename(path, split)

    data = None
    print(f"Loading {data_dir}")
    with data_dir.open() as f:
        data = json.load(f)
    return data