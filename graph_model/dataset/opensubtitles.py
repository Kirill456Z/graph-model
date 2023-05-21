import argparse
import os
import json
import re
import typing as tp
from pathlib import Path

from tqdm import tqdm
from sklearn.model_selection import train_test_split

OPENSUBTITLES_PATH = "/home/akhmadjonov/workspace/OpenSubtitles/en/lines"
OUTPUT_PATH = "/home/akhmadjonov/workspace/DialogGraphConstructing/opensubtitles"
NUMBER_OF_SPLITS = 2

def _parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line args."""

    def _positive_int(value):
        """Define a positive integer ArgumentParser type."""
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError(
                "Value must be positive, {} was passed.".format(value))
        return value

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sentence_files",
        default=OPENSUBTITLES_PATH,
        help="The path to the splitted dataset files folder")
    parser.add_argument(
        "--dialog_size",
        default=10,
        help="The maximum number of utterances in a dialogue.")
    parser.add_argument(
        "--min_len",
        default=9, type=_positive_int,
        help="The minimum length of an utterance.")
    parser.add_argument(
        "--max_len",
        default=127, type=_positive_int,
        help="The maximum length of an utterance.")
    parser.add_argument(
        "--output_dir", default=OUTPUT_PATH,
        help="Output directory to write the dataset.")
    parser.add_argument(
        "--train_split", default=0.9,
        type=float,
        help="The proportion of data to put in the training set.")
    parser.add_argument(
        "--num_shards_test", default=100,
        type=_positive_int,
        help="The number of shards for the test set.")
    parser.add_argument(
        "--num_shards_train", default=1000,
        type=_positive_int,
        help="The number of shards for the train set.")

    return parser.parse_args(argv)

def _concat_files(files: tp.List[str]) -> tp.List[str]:
    lines = []
    for file in files:
        with open(file, "r") as f:
            print(f"Reading file: {file}")
            lines.extend(f.readlines())
    return lines

def _should_skip(line: str, min_len: int, max_len: int) -> bool:
    return len(line) < min_len or len(line) > max_len

def _build_turn(turn_id: int, speaker: str, utterance: str) -> tp.Dict[str, any]:
    return {
        "turn_id": turn_id,
        "speaker": speaker,
        "utterance": utterance
    }

def _process_dialogue(dialogue_id: int, dialogue: tp.List[str]) -> tp.Dict[str, any]:
    result = {
        "dialogue_id": dialogue_id,
        "turns": []
    }
    speakers = ["A", "B"]
    for i, utt in enumerate(dialogue):
        result["turns"].append(_build_turn(i, speakers[i % 2], utt))
    return result

def _preprocess_line(line):
    # line = line.decode("utf-8")

    # Remove the first word if it is followed by colon (speaker names)
    # NOTE: this wont work if the speaker's name has more than one word
    line = re.sub('(?:^|(?:[.!?]\\s))(\\w+):', "", line)

    # Remove anything between brackets (corresponds to acoustic events).
    line = re.sub("[\\[(](.*?)[\\])]", "", line)

    # Strip blanks hyphens and line breaks
    line = line.strip(" -\n")

    return line

def _process_lines(lines: tp.List[str], args) -> tp.List[dict]:
    dialogues = []
    tmp_dialogue = []
    dialogue_id = 0
    for line in tqdm(lines):
        processed_line = _preprocess_line(line)
        if _should_skip(processed_line, args.min_len, args.max_len):
            continue
        tmp_dialogue.append(processed_line)
        if len(tmp_dialogue) >= args.dialog_size:
            dialogues.append(_process_dialogue(dialogue_id, tmp_dialogue))
            tmp_dialogue.clear()
            dialogue_id += 1

    return dialogues

def _split_dataset(dialogues: tp.List[dict], train_proportion: float) -> tp.Tuple[list]:
    train, test = train_test_split(dialogues, train_size=train_proportion)
    for i in range(len(train)):
        train[i]["dialogue_id"] = i
    
    for i in range(len(test)):
        test[i]["dialogue_id"] = i
    
    print(f"Train size: {len(train)}, test size: {len(test)}")
    return train, test

def validate_filename(path: str, split: str) -> tp.Tuple[Path]:
    if split not in ["train", "test"]:
        raise ValueError(
            'Invalid split type. Appropriate splits: ["train", "test"]'
        )
    extension = ".json"

    filename = "".join([split, extension])

    data_dir = Path(path) / filename

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Given file: {data_dir} doesn't exist.")

    return filename, data_dir

def load_opensubtitles(path: str, split: str) -> tp.List[dict]:
    _, data_dir = validate_filename(path, split)
    data = None
    print(f"Loading {data_dir}")
    with data_dir.open() as f:
        data = json.load(f)
    return data

def run(argv=None):
    args = _parse_args(argv)
    raw_data = _concat_files(list(Path(args.sentence_files).iterdir())[:NUMBER_OF_SPLITS])
    print("Processing files...")
    dialogues = _process_lines(raw_data, args)
    print("Splitting to train and test...")
    train, test = _split_dataset(dialogues, args.train_split)
    train_output_file = "train.json"
    test_output_file = "test.json"
    with open(f"{args.output_dir}/{train_output_file}", "w+") as f:
        print(f"Writing train to {args.output_dir}/{train_output_file}")
        json.dump(train, f, indent=2)

    with open(f"{args.output_dir}/{test_output_file}", "w+") as f:
        print(f"Writing test part to {args.output_dir}/{test_output_file}")
        json.dump(test, f, indent=2)

if __name__ == "__main__":
    run()
