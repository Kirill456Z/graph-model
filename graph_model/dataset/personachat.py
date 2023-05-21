# !wget http://parl.ai/downloads/personachat/personachat.tgz
# !tar xzvf personachat.tgz

import json
from pathlib import Path
import os
import typing as tp

from tqdm import tqdm
import re

def validate_filename(
    path: str,
    split: str,
    persona_type: str = "both",
    is_original: bool = True,
    is_preprocessed: bool = True,
):
    if split not in ["train", "val", "test"]:
        raise ValueError(
            'Invalid split type. Appropriate splits: ["train", "val", "test"]'
        )

    if persona_type not in ["both", "self", "their", "none"]:
        raise ValueError(
            'Persona type must be one of ["both", "self", "their", "none"]'
        )

    split_prefix = "valid" if split == "val" else split
    persona_prefix = "other" if persona_type == "their" else persona_type
    file_suffix = "original" if is_original else "revised"
    extension = ".json" if is_preprocessed else ".txt"

    filename = "".join([split_prefix, "_", persona_prefix, "_", file_suffix, extension])

    data_dir = Path(path) / filename

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Given file: {data_dir} doesn't exist.")

    return filename, data_dir


def collect_personas(dialogue: tp.List[str]) -> tp.Tuple[int, dict]:
    result = {"own": "", "their": ""}
    curr_persona_type = ""
    i = 0
    while True:
        splitted = dialogue[i].split("your persona:")
        curr_persona_type = "own"
        if len(splitted) == 1:
            splitted = dialogue[i].split("partner's persona:")
            curr_persona_type = "their"
            if len(splitted) == 1:
                break
        result[curr_persona_type] = "".join([result[curr_persona_type], splitted[1]])
        i += 1
    return i, result


def build_turn(id: int, speaker: str, utterance: "str"):
    return {
        "turn_id": id,
        "speaker": speaker,
        "utterance": utterance,
    }


def process_dialogue(
    dialogue_id: int, dialogue: tp.List[str], persona_type: str = "both"
) -> tp.Dict[str, tp.Any]:
    result = {"dialogue_id": dialogue_id, "PERSONA 1": "", "PERSONA 2": "", "turns": []}
    i = 0
    if persona_type != "none":
        i, collected_personas = collect_personas(dialogue)
        result["PERSONA 2"] = collected_personas["own"]
        result["PERSONA 1"] = collected_personas["their"]
    utterance_id = 0
    while i < len(dialogue):
        utt_a, utt_b = dialogue[i].split("\t")[:2]
        utt_a = re.split("^[0-9]+", utt_a)[1]
        result["turns"].extend(
            [
                build_turn(utterance_id, "PERSON 1", utt_a),
                build_turn(utterance_id + 1, "PERSON 2", utt_b),
            ]
        )
        utterance_id += 2
        i += 1
    return result


def process_data(
    path: str, split: str, persona_type: str = "both", is_original: bool = True
):
    """
    Preprocess dataset and convert it to json.

    path - path to the dataset
    split - "test" / "val" / "train"
    persona_type - "none" / "self" / "their" / "both". Default: "both"
    is_original - True if original persona is needed, otherwise revised persona is used. Default: True
    """
    filename, data_dir = validate_filename(
        path, split, persona_type, is_original, False
    )
    new_data_dir = data_dir.parent / "processed" / Path(filename).with_suffix(".json")
    data = []
    print(f"Preprocessing {data_dir}")
    is_first_line = True
    with data_dir.open() as f:
        lines = f.readlines()
        slice = []
        dialogue_idx = 0
        for line in tqdm(lines):
            line = line.strip()
            row_num = int(line.split()[0])
            if row_num == 1 and not is_first_line:
                data.append(process_dialogue(dialogue_idx, slice, persona_type))
                slice.clear()
                dialogue_idx += 1
            elif row_num == 1:
                is_first_line = False
            slice.append(line)
    with new_data_dir.open("w+") as f:
        print(f"Writing to {new_data_dir}")
        f.write(json.dumps(data, indent=2))


def load_personachat(
    path: str, split: str, persona_type: str = "both", is_original: bool = True
) -> tp.List[dict]:
    """
    Load preprocessed PersonaChat Dataset
    path - path to the dataset
    split - "test" / "val" / "train"
    persona_type - "none" / "self" / "their" / "both". Default: "both"
    is_original - True if original persona is needed, otherwise revised persona is used. Default: True
    """
    _, data_dir = validate_filename(path, split, persona_type, is_original)

    data = None
    print(f"Loading {data_dir}")
    with data_dir.open() as f:
        data = json.load(f)
    return data
