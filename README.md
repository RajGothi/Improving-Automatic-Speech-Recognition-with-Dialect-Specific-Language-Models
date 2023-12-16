## Abstract
We present an end-to-end Automatic Speech Recognition (ASR) system in the context of the recent challenge tasks for Bhojpuri and Bengali. Our implementation follows the currently popular wav2vec models while we investigate ways to leverage the dialect-categorised data in order to improve ASR performance. We report overall improvements in word error rate with dialect-specific language models for each of the languages. We present an analysis that provides insights into some of the factors underlying the success of dialect-specific language models.


## Requirements
```bash
pip install .
wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2
```

## Preprocessing

To get the speech and corresponding transcript from the dataset and convert into the Huggigface-Dataset format:
```bash
python pre_processing.py \
  --repo_name "Trained_Model/wav2vec2-bh" \
  --path "Dataset/bh" \
  --dev_path "Dataset/bh/bh_dev/dev" \
  --train_text_path "MADASR-Competition/RESPIN_ASRU_Challenge_2023/corpus/bh/train/text" \
  --dev_text_path "MADASR-Competition/RESPIN_ASRU_Challenge_2023/corpus/bh/dev/text" \
  --dataset_name "bh_processed" \
  --vocab_path "Vocab/vocab_bh.json"
```

To Create the Dialect ID based Dataset for Evaluation:
```bash
python eval_dialect_dataset.py \
  --dev_path "/home/raj/Lab/Dataset/bh/bh_dev/dev" \
  --utt2lang_path "RESPIN_ASRU_Challenge_2023/corpus/bh/dev/utt2lang" \
  --dev_text_path "RESPIN_ASRU_Challenge_2023/corpus/bh/dev/text" \
  --processor_name "Trained_Model/wav2vec2-bh" \
  --dataset_name "Dataset/bh_dev_dialect"
```


## Acoustic Model (AM) Training
```bash
python train_AM.py --config_path="Config/train_AM.yaml"
```

## Language Model (LM) Training
```bash
python train_LM-All.py --config_path="Config/train_LM-All.yaml"
python train_LM_Dialect.py --config_path='Config/train_LM-Dialect.yaml'
```

## Evaluation
```bash
python test_AM.py --config_path="Config/test_AM.yaml"
python test_AM_LM-All.py --config_path="Config/test_AM_LM-All.yaml"
python test_AM_LM-Dialect.py --config_path="Config/test_AM_LM-Dialect.yaml"
```

## Inference
To run the model for one audio, you can use the notebook below.

 [Notebook](Notebook/inference.ipynb)
