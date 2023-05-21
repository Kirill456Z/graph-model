# Steps on how to get and prepare AmazonQA dataset:
# --> ! wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_*.json.gz -- * means the name of the 
# --> ! gzip -d qa_*.json.gz
# --> convert each json file to strict json format and delete the original json files
# --> merge converted json files
# --> preprocess the merged file
# --> split to train and test sets

import argparse
import os
import json
import re
import typing as tp
from pathlib import Path

from tqdm import tqdm
from sklearn.model_selection import train_test_split

AMAZONQA_PATH = "/home/akhmadjonov/workspace/DialogGraphConstructing/amazonqa"

def split_dataset(input_file: str, output_dir: str = None, train_size: float = 0.9):
    if not output_dir:
        output_dir = Path(os.path.dirname(input_file)) / "processed"
    
    test_file = output_dir / "test.json"
    train_file = output_dir / "train.json"

    data = None
    with open(input_file, "r") as f:
        data = json.load(f)
    
    train, test = train_test_split(data, train_size=train_size)
    for i in range(len(train)):
        train[i]["dialogue_id"] = i
    
    for i in range(len(test)):
        test[i]["dialogue_id"] = i
    
    print(f"Train size: {len(train)}, test size: {len(test)}")
    
    with open(test_file, "w") as f:
        json.dump(test, f, indent=2)

    with open(train_file, "w") as f:
        json.dump(train, f, indent=2)

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

def load_amazonqa(path: str, split: str) -> tp.List[dict]:
    _, data_dir = validate_filename(path, split)
    data = None
    print(f"Loading {data_dir}")
    with data_dir.open() as f:
        data = json.load(f)
    return data

def convert_to_strict_json(input_file: str, output_file: str = None):
    if not output_file:
        output_file = Path(os.path.dirname(input_file)) / f"strict_{os.path.basename(input_file)}"
    lines = None

    with open(f"{input_file}", "r+") as f:
        print(f"Reading from {input_file} ...")
        lines = f.readlines()
    
    print("Processing...")
    json_data = []
    for line in tqdm(lines):
        json_data.append(eval(line))

    with open(f"{output_file}", "w+") as f:
        print(f"Writing to {output_file} ...")
        json.dump(json_data, f, indent=2)

def merge_files(datapath: str, output_dir: str = None):
    if not output_dir:
        output_dir = datapath
    output_file = Path(output_dir) / "merged.json"
    merged_data = []
    for file in Path(datapath).iterdir():
        if os.path.basename(file) == "merged.json" or file.suffix != ".json":
            continue
        with open(file, "r") as f:
            merged_data = [*merged_data, *json.load(f)]
    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=2)



def preprocess_data(data_file: str, output_dir: str = None):
    if not output_dir:
        output_dir = os.path.dirname(data_file)
    output_file = Path(output_dir) / "processed.json"
    data = None
    with open(data_file, "r") as f:
        print(f"Reading {data_file} ...")
        data = json.load(f)
    
    new_data = []
    print("Processing...")
    for item in tqdm(data):
        if len(item["answer"].split(' ')) < 4:
            continue
        new_data.append(item)
    
    with open(output_file, "w") as f:
        print(f"Writing to {output_file} ...")
        json.dump(new_data, f, indent=2)