import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_from_disk, load_metric,Dataset,DatasetDict,concatenate_datasets
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config,Wav2Vec2ProcessorWithLM,AutoProcessor
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import TrainingArguments, Trainer
from pyctcdecode import build_ctcdecoder
import json
from datasets import Audio
from IPython.display import display, HTML
import pandas as pd
import numpy as np
import gc
import wandb
import random
import re
import json
import os

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
d1_repo_name = "6_d1_wav2vec2-3-4-5-fine-tune-additional_train_bhojpuri"
d2_repo_name = "6_d2_wav2vec2-3-4-5-fine-tune-additional_train_bhojpuri"
d3_repo_name = "6_d3_wav2vec2-3-4-5-fine-tune-additional_train_bhojpuri"

repo_name = " RajGothi-DAP_IITB_submission-1_bh_track-3.json"

# torch.cuda.set_device(0)
# torch.cuda.current_device()
# os.environ['CUDA_VISIBLE_DEVICES']='0'
gc.collect()
torch.cuda.empty_cache()
wandb.init(
    project="Bhojpuri_LM",
    entity = "rajgothi6"
)
json_path = repo_name

def save_elements(dataset):
    # print(dataset)

    dataset_from_pandas = Dataset.to_pandas(dataset)
    json_data = dataset_from_pandas.to_dict(orient='records')

    with open(repo_name, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False)
    
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = model(input_values).logits

#   pred_ids = torch.argmax(logits, dim=-1)
#   batch["pred_str"] = processor.batch_decode(pred_ids)[0]

    if(batch['lang_id']=="D1"):
        processor = d1_processor
    if(batch['lang_id']=="D2"):
        processor = d2_processor
    if(batch['lang_id']=="D3"):
        processor = d3_processor

    batch["hypothesis"] = processor.batch_decode(logits.cpu().numpy())[0]
    batch["hypothesis"]= ''.join(batch["hypothesis"])
    return batch

if __name__ == "__main__":
    # argparser

    # -----------------------------------------------------------------------------------------------------------------
    #To create LM folder:
    # processor = AutoProcessor.from_pretrained("wav2vec2-latest-1-4-5-fine-tune-bengali")
    # tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("wav2vec2-latest-1-4-5-fine-tune-bengali", eos_token=None, bos_token=None,unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    # vocab_dict = tokenizer.get_vocab()
    # sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

    # decoder = build_ctcdecoder(
    #     labels=list(sorted_vocab_dict.keys()),
    #     kenlm_model_path="/home/raj/MADASR-Competition/5gram_correct.arpa",
    # )

    # processor_with_lm = Wav2Vec2ProcessorWithLM(
    #     feature_extractor=processor.feature_extractor,
    #     tokenizer=tokenizer,
    #     decoder=decoder
    # )

    # processor_with_lm.save_pretrained("wav2vec2-latest-1-4-5-LM-fine-tune-bengali")
    # ----------------------------------------------------------------------------------------------------------------


    dataset = load_from_disk("test_dataset") 
    # dataset = dataset[:5]
    # dataset = Dataset.from_dict(dataset)
    # print(dataset)
    d1_processor = Wav2Vec2ProcessorWithLM.from_pretrained(d1_repo_name)
    d2_processor = Wav2Vec2ProcessorWithLM.from_pretrained(d2_repo_name)
    d3_processor = Wav2Vec2ProcessorWithLM.from_pretrained(d3_repo_name)
    

    model = Wav2Vec2ForCTC.from_pretrained("wav2vec2-3-4-5-fine-tune-bhojpuri")

    processor_decode = Wav2Vec2Processor.from_pretrained("wav2vec2-3-4-5-fine-tune-bhojpuri")

    wer_metric = load_metric("wer")

    #if you have already stored the pre-trained model...
    # processor = Wav2Vec2Processor.from_pretrained(repo_name)
    # model = Wav2Vec2ForCTC.from_pretrained(
    #     repo_name)

    model.cuda()
    # model.freeze_feature_encoder()

    #training the model
    # gc.collect()
    # torch.cuda.empty_cache()

    #evaluate the model on test dataset (currently we only have dev set)
    results =dataset.map(map_to_result, remove_columns=['lang_id','input_values','input_length'])
    # print("Test WER: {:.4f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))
    print("completed")
    save_elements(results)