import os
import random
from datasets import Dataset, DatasetDict
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
import re
import json
import numpy as np
from datasets import Audio
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
import argparse


chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'


def read_text_file(path):
    data = []
    dict= {}
    with open(path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()  # Remove leading/trailing whitespace and newline characters
            first_space = line.find(" ")
            audio_info = line[:first_space].split('_')
            label = line[first_space+1:]
            spkids = audio_info[0]
            text_id = audio_info[1]
            recording_id = audio_info[2] 
            dict[recording_id] = [spkids,text_id,recording_id,label]
            data.append([spkids,text_id,recording_id,label])
    return data,dict

def convert_list_to_dataset_HF(dataset):
    data_dict = {
    "spkids": [item[0] for item in dataset],
    "text_id": [item[1] for item in dataset],
    "path" : [item[2] for item in dataset],
    "label" : [item[3] for item in dataset]
    }
    return Dataset.from_dict(data_dict)

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))


def remove_special_characters(batch):
    # batch["label"] = re.sub(chars_to_ignore_regex, '', batch["label"]).lower() + " "
    batch["label"] = re.sub(chars_to_ignore_regex, '', batch["label"]) + " "
    return batch

def extract_all_chars(batch):
  all_text = " ".join(batch["label"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset(batch):
    audio = batch["path"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["label"]).input_ids
    return batch

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process audio files.")
    parser.add_argument("--repo_name", type=str, default="Trained_Model/wav2vec2-bh", help="Repository name")
    parser.add_argument("--path", type=str, default="/home/raj/Lab/Dataset/bh", help="Language Dataset folder path")
    parser.add_argument("--dev_path", type=str, default="/home/raj/Lab/Dataset/bh/bh_dev/dev", help="Dev folder path")
    parser.add_argument("--train_text_path", type=str, default="RESPIN_ASRU_Challenge_2023/corpus/bh/train/text", help="Train text file path")
    parser.add_argument("--dev_text_path", type=str, default="RESPIN_ASRU_Challenge_2023/corpus/bh/dev/text", help="Dev text file path")
    parser.add_argument("--dataset_name", type=str, default="Dataset/bh_processed", help="Dataset name")
    parser.add_argument("--vocab_path", type=str, default="Vocab/vocab_bh.json", help="Vocabulary file path")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    repo_name = args.repo_name
    path = args.path
    dev_path = args.dev_path
    train_text_path = args.train_text_path
    dev_text_path = args.dev_text_path
    dataset_name = args.dataset_name
    vocab_path = args.vocab_path
    
    dataset = os.listdir(path)

    train_files = []
    for split in dataset:
        if 'train' in split:
            dataset_path = f"{path}/{split}"
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith(('.mp3', '.wav', '.flac')):
                        train_files.append(os.path.join(root, file))

    dev_files = os.listdir(dev_path)

    print('Total Train audios: ',len(train_files))
    print('Total Dev audios: ',len(dev_files))

    train_data,train_dict = read_text_file(train_text_path)

    dev_data,dev_dict = read_text_file(dev_text_path)


    selected_train_dataset = []
    for name in train_files:
        last_ind = name.rfind('/')
        audio_info = train_dict[name[last_ind+1:-4]]
        audio_info[2] = name #set path instead of recording id now...
        selected_train_dataset.append(audio_info)

    selected_dev_dataset = []
    for name in dev_files:
        audio_info = dev_dict[name[:-4]]
        audio_info[2] = os.path.join(dev_path,name) #set path instead of recording id now...
        selected_dev_dataset.append(audio_info)

    train_data_hf = convert_list_to_dataset_HF(selected_train_dataset)
    dev_data_hf = convert_list_to_dataset_HF(selected_dev_dataset)

    dataset = DatasetDict({"train": train_data_hf, "dev": dev_data_hf})

    show_random_elements(dataset["train"].remove_columns(["spkids", "text_id","path"]), num_examples=10)

    dataset = dataset.map(remove_special_characters)


    # Create the Vocab file:
    vocabs = dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset.column_names["train"])

    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["dev"]["vocab"][0]))

    vocab_list.sort()

    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(vocab_path, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(repo_name)

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=64)

    #save locally:
    dataset.save_to_disk(dataset_name) 

