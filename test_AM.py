import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_from_disk, load_metric,Dataset,DatasetDict,concatenate_datasets
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import TrainingArguments, Trainer
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
import argparse
import yaml

# wandb.init(
#     project="MADASR",
#     entity = "account"
# )

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    # print(df)
    
    df = pd.DataFrame(dataset)
    df.to_csv(csv_path)
    # display(HTML(df.to_html()))

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


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def map_to_result(batch):
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0]

  batch["text"] = processor.decode(batch["labels"], group_tokens=False)
  
  return batch

def parse_args():
    parser = argparse.ArgumentParser(description="Train the LM model")
    parser.add_argument("--config_path", type=str, help="Path to the YAML configuration file")
    return parser.parse_args()

if __name__ == "__main__":
    # argparser

    args = parse_args()

    # Load configuration from the specified file
    with open(args.config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    repo_name = config['repo_name']
    csv_path = config['csv_path']
    dataset_path = config['dataset_path']

    dataset = load_from_disk(dataset_path)
    dataset = dataset['dev'] 

    processor = Wav2Vec2Processor.from_pretrained(repo_name)

    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True)

    wer_metric = load_metric("wer")

    #if you have already stored the pre-trained model...
    model = Wav2Vec2ForCTC.from_pretrained(repo_name)

    model.cuda()

    results = dataset.map(map_to_result, remove_columns=dataset.column_names)
    print("Test WER: {:.4f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))

    show_random_elements(results)