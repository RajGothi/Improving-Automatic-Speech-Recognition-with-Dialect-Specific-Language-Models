import re
import subprocess
from transformers import AutoProcessor,Wav2Vec2CTCTokenizer
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2ProcessorWithLM
from transformers import Wav2Vec2CTCTokenizer
import json
import argparse
import yaml

chars_to_ignore_regex = '[\,\?\.\!\-\;\:()[]|}{।\"]'

def create_kenlm(i):

    command = f'! kenlm/build/bin/lmplz -o {n_gram} <"Language_model/d{i}_text_{lang_code}.txt" > "Language_model/d{i}_{n_gram}_gram_additional_train_{lang_code}.arpa"'

    # Run the command in the shell
    process = subprocess.Popen(command, shell=True)
    process.wait()

    with open(f"Language_model/d{i}_{n_gram}_gram_additional_train_{lang_code}.arpa", "r") as read_file, open(f"Language_model/d{i}_{n_gram}_gram_correct_additional_train_{lang_code}.arpa", "w") as write_file:
        has_added_eos = False
        for line in read_file:
            if not has_added_eos and "ngram 1=" in line:
                count=line.strip().split("=")[-1]
                write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
            elif not has_added_eos and "<s>" in line:
                write_file.write(line)
                write_file.write(line.replace("<s>", "</s>"))
                has_added_eos = True
            else:
                write_file.write(line)

def remove_pun(text):
    text = re.sub(chars_to_ignore_regex, ' ', text)
    text = re.sub("incomplete",'',text)
    text = re.sub("INCOMPLETE",'',text)
    text = re.sub("Incomplete",'',text)
    return text

def read_text_file(path,d1_text,d2_text,d3_text,d4_text,d5_text):
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
            text = label + "\n"

            if(lang_id=="D1"):
                d1_text += text
            if(lang_id == "D2"):
                d2_text += text
            if(lang_id == "D3"):
                d3_text += text
            if(lang_id == "D4"):
                d4_text += text
            if(lang_id == "D5"):
                d5_text += text

            dict[recording_id] = [spkids,text_id,recording_id,label]
            data.append([spkids,text_id,recording_id,label])
    return data,dict,d1_text,d2_text,d3_text,d4_text,d5_text

def parse_args():
    parser = argparse.ArgumentParser(description="Train the LM model")
    parser.add_argument("--config_path", type=str, help="Path to the YAML configuration file")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Load configuration from the specified file
    with open(args.config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    lang_code = config['lang_code']
    total_dialect = config['total_dialect']
    n_gram = config['n_gram']
    output_LM_name = config['output_LM_name']
    model_name = config['model_name']
    train_text_path = config['train_text_path']
    utt2lang_path = config['utt2lang_path']
    additional_corpus_path = config['additional_corpus_path']

    d1_text = ""
    d2_text = ""
    d3_text = ""
    d4_text = ""
    d5_text = ""
    text = ""

    with open(additional_corpus_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()  # Remove leading/trailing whitespace and newline characters
            first_space = line.find("\t")
            lang = line[:first_space].split('_')[0]
            text = line[first_space+1:] + "\n"        
            if(lang=="D1"):
                d1_text += text
            if(lang == "D2"):
                d2_text += text
            if(lang == "D3"):
                d3_text += text
            if(lang == "D4"):
                d4_text += text
            if(lang == "D5"):
                d5_text += text
                
    d1_text = remove_pun(d1_text)
    d2_text = remove_pun(d2_text)
    d3_text = remove_pun(d3_text)
    d4_text = remove_pun(d4_text)
    d5_text = remove_pun(d5_text)


    utt2lang = {}
    with open(utt2lang_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            first_space = line.find("\t")
            audio_id = line[:first_space]
            lang = line[first_space+1:]
            utt2lang[audio_id] = lang


    train_data,train_dict,d1_text,d2_text,d3_text,d4_text,d5_text = read_text_file(train_text_path,d1_text,d2_text,d3_text,d4_text,d5_text)

    unique_text = d1_text.split('\n')
    unique_text = set(unique_text)
    d1_text_unique = '\n'.join(unique_text)

    unique_text = d2_text.split('\n')
    unique_text = set(unique_text)
    d2_text_unique = '\n'.join(unique_text)

    unique_text = d3_text.split('\n')
    unique_text = set(unique_text)
    d3_text_unique = '\n'.join(unique_text)

    unique_text = d4_text.split('\n')
    unique_text = set(unique_text)
    d4_text_unique = '\n'.join(unique_text)

    unique_text = d5_text.split('\n')
    unique_text = set(unique_text)
    d5_text_unique = '\n'.join(unique_text)

    with open(f"Language_model/d1_text_{lang_code}.txt", "w") as file:
        file.write(d1_text_unique)

    with open(f"Language_model/d2_text_{lang_code}.txt", "w") as file:
        file.write(d2_text_unique)

    with open(f"Language_model/d3_text_{lang_code}.txt", "w") as file:
        file.write(d3_text_unique)

    if(lang_code=='bn'):
        with open(f"Language_model/d4_text_{lang_code}.txt", "w") as file:
            file.write(d4_text_unique)

        with open(f"Language_model/d5_text_{lang_code}.txt", "w") as file:
            file.write(d5_text_unique)
        
    for i in range(1,total_dialect+1):
        create_kenlm(i)

    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name, eos_token=None, bos_token=None,unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    vocab_dict = tokenizer.get_vocab()
    sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

    for i in range(1,total_dialect+1):
        decoder = build_ctcdecoder(
            labels=list(sorted_vocab_dict.keys()),
            kenlm_model_path=f"Language_model/d{i}_{n_gram}_gram_correct_additional_train_{lang_code}.arpa",
        )
        processor_with_lm = Wav2Vec2ProcessorWithLM(
            feature_extractor=processor.feature_extractor,
            tokenizer=tokenizer,
            decoder=decoder
        )

        processor_with_lm.save_pretrained(f"Language_model/d{i}_{n_gram}_{output_LM_name}")
