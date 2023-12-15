import re
import subprocess
from transformers import AutoProcessor,Wav2Vec2CTCTokenizer
from pyctcdecode import build_ctcdecoder
import json
from transformers import Wav2Vec2ProcessorWithLM
from transformers import Wav2Vec2CTCTokenizer
import yaml
import argparse

def read_text_file(path,text):
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
            text += label + "\n"
            dict[recording_id] = [spkids,text_id,recording_id,label]
            data.append([spkids,text_id,recording_id,label])
    return data,dict,text


def create_kenlm():

    command = f'kenlm/build/bin/lmplz -o {n_gram} < {output_text_path} > "Language_model/{n_gram}gram_additional_train_{lang_code}.arpa"'

    # Run the command in the shell
    process = subprocess.Popen(command, shell=True)
    process.wait()

    with open(f"Language_model/{n_gram}gram_additional_train_{lang_code}.arpa", "r") as read_file, open(f"Language_model/{n_gram}gram_correct_additional_train_{lang_code}.arpa", "w") as write_file:
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

def parse_args():
    parser = argparse.ArgumentParser(description="Train the LM model")
    parser.add_argument("--config_path", type=str, help="Path to the YAML configuration file")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Load configuration from the specified file
    with open(args.config_path, "r") as config_file:
        config = yaml.safe_load(config_file)


    text = ""
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:()[]|}{।\"]'

    additional_corpus_path = config['additional_corpus_path']
    train_text_path = config['train_text_path']
    output_text_path = config['output_text_path']
    lang_code = config['lang_code']
    model_name = config['model_name']
    output_LM_name = config['output_LM_name']
    n_gram = config['n_gram']

    with open(additional_corpus_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()  # Remove leading/trailing whitespace and newline characters
            first_space = line.find("\t")
            text += line[first_space+1:] + "\n"        

    text = re.sub(chars_to_ignore_regex, ' ', text)
    text = re.sub("incomplete",'',text)
    text = re.sub("INCOMPLETE",'',text)
    text = re.sub("Incomplete",'',text)


    train_data,train_dict,text = read_text_file(train_text_path,text)


    unique_text = text.split('\n')
    # print(len(unique_text))
    unique_text = set(unique_text)
    # print(len(unique_text))
    unique_text = '\n'.join(unique_text)

    with open(output_text_path, "w") as file:
        file.write(unique_text)

    # If you have not build kenlm then clone kenlm repo and run below command on command line. 
    # !wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
    # ! mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2
    # ! ls kenlm/build/bin

    create_kenlm()

    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name, eos_token=None, bos_token=None,unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    vocab_dict = tokenizer.get_vocab()
    sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path=f"Language_model/{n_gram}gram_correct_additional_train_{lang_code}.arpa",
    )

    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=tokenizer,
        decoder=decoder
    )

    processor_with_lm.save_pretrained(output_LM_name)