import csv

import torch
import os
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from tqdm import tqdm


id2label = {0: "no_symptom", 1: "has_symptom"}
label2id = {"no_symptom": 0, "has_symptom": 1}


# checkpoint = "model/cardiffnlp/twitter-roberta-large-topic-sentiment-latest/classification-2/lr-1e-7-downsample-1/checkpoint-108200"
checkpoint = "model/cardiffnlp/twitter-roberta-large-topic-sentiment-latest/ade_merged_classification/gpt4-lr-1e-07-downsample-1-max_len-50/checkpoint-16260"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

# dev_dataset_dir = 'data/dev'
# dataset_type = 'ade_merged_classification'
# with open(os.path.join(dev_dataset_dir, dataset_type, 'has_symptom.txt')) as f:
#     dev_has_symptom_list = f.readlines()

classified_location = 'data/train/classified.csv'
# classified_location = 'data/test/classified.csv'
# classified_location = 'data/dev/classified.csv'
classified_list = []

with open("data/train/ade_merged.json", "r") as f:
# with open("data/test/ade_merged.json", "r") as f:
# with open("data/dev/ade_merged.json", "r") as f:
# with open("data/dev/tweet_span_classification_prefix/tweet.json", "r") as f:
    tweet_obj = json.load(f)
    # has_symptom_list = [s.strip() for s in tweet_obj.values()]
    # dev_has_symptom_list = [s.strip() for s in dev_has_symptom_list]
    # for tweet in dev_has_symptom_list:
    #     if tweet.strip() not in has_symptom_list:
    #         raise Exception(tweet)

    for tweet_id, text in tqdm(tweet_obj.items()):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        if not predicted_class_id:
            continue

        classified_list.append({
            'tweet_id': tweet_id,
            'text': text,
            # 'ground-truth': 1 if text.strip() in dev_has_symptom_list else 0,
            'predicted': predicted_class_id
        })

with open(classified_location, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['tweet_id', 'text', 'predicted'])
    # writer = csv.DictWriter(f, fieldnames=['tweet_id', 'text', 'ground-truth', 'predicted'])
    writer.writeheader()
    for line in classified_list:
        writer.writerow(line)
