import csv

import torch
import os
import json
import numpy as np

from datasets import Dataset, load_from_disk, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers import pipeline
from tqdm import tqdm
from nltk.tokenize.treebank import TreebankWordDetokenizer


id2label = {0: "O", 1: "B-ADE", 2: "I-ADE"}
label2id = {"O": 0, "B-ADE": 1, "I-ADE": 2}

checkpoint = "model/Clinical-AI-Apollo/Medical-NER/tweet_span_ade_ner_classified/gpt4-lr-2e-05-downsample-1-max_len-70-ignore-bio/checkpoint-38246"
# checkpoint = "model/Clinical-AI-Apollo/Medical-NER/tweet_span_ade_ner/gpt4-lr-2e-05-downsample-1-max_len-70-ignore-bio/checkpoint-15080"
# checkpoint = "model/cardiffnlp/twitter-roberta-base-ner7-latest/tweet_ade_ner/gpt4-lr-2e-05-downsample-1-max_len-50-ignore-bio/checkpoint-24024"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForTokenClassification.from_pretrained(
    checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

MAX_LEN = 70
predicted_location = 'data/test/ner_prediction.json'
classified_location = 'data/test/classified.csv'
# predicted_location = 'data/dev/ner_prediction.json'
# classified_location = 'data/dev/classified.csv'
predicted_dict = dict()

# dataset = load_from_disk('data/2024_span_ner_gpt4')
# dataset = load_from_disk('data/2024_ade_ner_gpt4')
# dataset = load_from_disk('data/2024_summary_ner_gpt4')
# dataset = load_from_disk('data/2024_ner')

# dev_dataset = dataset['dev']
#
# with open(classified_location, 'r') as f:
#     reader = csv.DictReader(f)
#     predicted_tweets = []
#     for line in reader:
#         if not int(line['predicted']):
#             continue
#         predicted_tweets.append(line['tweet_id'])

with open('data/test/span_merged.json', 'r') as f:
    dataset = json.load(f)


# for examples in dev_dataset:
#     tweet_id = examples['idx']
#     if tweet_id not in predicted_tweets:
#         continue

for tweet_id, text in dataset.items():
#     if tweet_id != 'SMM4H2022UAvDTQWOIacvBkzp':
    #     continue

    classifier = pipeline("ner", model=checkpoint)
    detokenizer = TreebankWordDetokenizer()
    # text = detokenizer.detokenize(examples["tokens"])
    classified = classifier(text)
    sep_index = text.index('[sep]')

    classified = [c for c in classified if c['start'] < sep_index]
    classified = sorted(classified, key=lambda x: x['index'])

    span_list = []
    current_span = []
    # tokenizer_special_token = 'Ġ'
    tokenizer_special_token = '▁'
    for c in classified:
        if c['entity'] == 'B-ADE' and len(current_span) and c['word'].startswith(tokenizer_special_token):
            span_list.append(detokenizer.detokenize(current_span))
            current_span = []

        if c['word'].startswith('%s' % tokenizer_special_token):
            current_span.append(c['word'].replace(tokenizer_special_token, ''))
        elif len(current_span):
            current_span[-1] += c['word']
        else:
            current_span.append(c['word'].replace(tokenizer_special_token, ''))

    if len(current_span):
        span_list.append(detokenizer.detokenize(current_span))
    # inputs = tokenizer(examples["tokens"], max_length=MAX_LEN, truncation=True, padding='max_length', is_split_into_words=True, return_tensors="pt")
    # with torch.no_grad():
    #     logits = model(**inputs).logits

    # predicted_dict[tweet_id] = {
    #     'prediction': np.argmax(logits, axis=2).squeeze().tolist(),
    #     'tokens': examples["tokens"]
    # }
    predicted_dict[tweet_id] = span_list

with open(predicted_location, 'w') as f:
    json.dump(predicted_dict, f)


print(f'Total: {len(predicted_dict)}')
