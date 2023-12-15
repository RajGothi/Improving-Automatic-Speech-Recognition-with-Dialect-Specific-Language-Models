from datasets import load_from_disk
import os
import random
from datasets import Dataset, DatasetDict
from datasets import Audio
from transformers import Wav2Vec2Processor
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Your script description")
    
    parser.add_argument("--dev_path", type=str, default="/home/raj/Lab/Dataset/bh/bh_dev/dev",
                        help="Path to the development dataset")
    
    parser.add_argument("--utt2lang_path", type=str, default="RESPIN_ASRU_Challenge_2023/corpus/bh/dev/utt2lang",
                        help="Path to the utt2lang file")
    
    parser.add_argument("--dev_text_path", type=str, default="RESPIN_ASRU_Challenge_2023/corpus/bh/dev/text",
                        help="Path to the development text file")
    
    parser.add_argument("--processor_name", type=str, default="Trained_Model/wav2vec2-bh",
                        help="Name or path of the processor")
    
    parser.add_argument("--dataset_name", type=str, default="Dataset/bh_dev_dialect",
                        help="Name or path of the dataset")

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    dev_path = args.dev_path
    utt2lang_path = args.utt2lang_path
    dev_text_path = args.dev_text_path
    processor_name = args.processor_name
    dataset_name = args.dataset_name

    dev_files = os.listdir(dev_path)

    utt2lang = {}
    with open(utt2lang_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            first_space = line.find("\t")
            audio_id = line[:first_space]
            lang = line[first_space+1:]
            utt2lang[audio_id] = lang

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
                lang_id = utt2lang[line[:first_space]]
                dict[recording_id] = [spkids,text_id,recording_id,label,lang_id]
                data.append([spkids,text_id,recording_id,label,lang_id])
        return data,dict

    def convert_list_to_dataset_HF(dataset):
        data_dict = {
        "spkids": [item[0] for item in dataset],
        "text_id": [item[1] for item in dataset],
        "path" : [item[2] for item in dataset],
        "label" : [item[3] for item in dataset],
        "lang_id" : [item[4] for item in dataset]
        }
        return Dataset.from_dict(data_dict)

    def prepare_dataset(batch):
        audio = batch["path"]

        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        
        with processor.as_target_processor():
            batch["labels"] = processor(batch["label"]).input_ids
            batch["lang_id"] = batch["lang_id"] 
        return batch


    dev_data,dev_dict = read_text_file(dev_text_path)

    selected_dev_dataset = []
    for name in dev_files:
        audio_info = dev_dict[name[:-4]]
        audio_info[2] = os.path.join(dev_path,name) #set path instead of recording id now...
        selected_dev_dataset.append(audio_info)

    dev_data_hf = convert_list_to_dataset_HF(selected_dev_dataset)

    dev_dataset = dev_data_hf.cast_column("path", Audio(sampling_rate=16000))

    processor = Wav2Vec2Processor.from_pretrained(processor_name)

    dev_dataset = dev_dataset.map(prepare_dataset,remove_columns=['spkids','text_id','path','label'] ,num_proc=4)

    dev_dataset.save_to_disk(dataset_name)