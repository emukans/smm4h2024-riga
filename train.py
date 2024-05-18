from random import shuffle, seed

import torch
import wandb
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, EarlyStoppingCallback

import os

import numpy as np

from datasets import Dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding


seed(42)
np.random.seed(42)



MAX_LEN = 50
batch_size = 64
epoch_count = 300
learning_rate = 2e-7
downsample_size = 0.8


# checkpoint = "distilbert/distilbert-base-uncased"
# checkpoint = "cardiffnlp/tweet-topic-21-multi"
# checkpoint = "cardiffnlp/twitter-roberta-base-sentiment-latest"
checkpoint = "cardiffnlp/twitter-roberta-large-topic-sentiment-latest"
# checkpoint = "cardiffnlp/twitter-roberta-large-hate-latest"
# checkpoint = "microsoft/Multilingual-MiniLM-L12-H384"
# checkpoint = "microsoft/deberta-v2-xxlarge-mnli"
# checkpoint = "Azie88/COVID_Vaccine_Tweet_sentiment_analysis_roberta"
# checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

train_dataset_dir = 'data/train'
dev_dataset_dir = 'data/dev'
dataset_type = 'ade_merged_classification'
# dataset_type = 'summary_merged_classification'

os.environ["WANDB_PROJECT"] = "smm4h2024-task1-tweet-classification"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_NAME"] = f"{checkpoint}/{dataset_type}/gpt4-lr-{learning_rate}-downsample-{downsample_size}-max_len-{MAX_LEN}"
# os.environ["WANDB_NOTES"] = "Spans extracted by GPT3.5 from tweets, classification. Downample 0.2"


id2label = {0: "no_symptom", 1: "has_symptom"}
label2id = {"no_symptom": 0, "has_symptom": 1}

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


with open(os.path.join(train_dataset_dir, dataset_type, 'has_symptom.txt')) as f:
    train_has_symptom_list = f.readlines()


with open(os.path.join(train_dataset_dir, dataset_type, 'no_symptom.txt')) as f:
    train_no_symptom_list = f.readlines()


shuffle(train_no_symptom_list)
train_no_symptom_list = train_no_symptom_list[:round(len(train_no_symptom_list) * downsample_size)]

with open(os.path.join(dev_dataset_dir, dataset_type, 'has_symptom.txt')) as f:
    dev_has_symptom_list = f.readlines()


with open(os.path.join(dev_dataset_dir, dataset_type, 'no_symptom.txt')) as f:
    dev_no_symptom_list = f.readlines()

wandb.init()
wandb.log({
    'train_size': len(train_has_symptom_list + train_no_symptom_list),
    'dev_size': len(dev_has_symptom_list + dev_no_symptom_list),
    'downsample_size': downsample_size,
    'train_has_symptom_proportion': len(train_has_symptom_list) / len(train_has_symptom_list + train_no_symptom_list),
    'dev_has_symptom_proportion': len(dev_has_symptom_list) / len(dev_has_symptom_list + dev_no_symptom_list),
    'model_size': model.num_parameters(),
    'max_len': MAX_LEN,
})

train_dataset = Dataset.from_dict({'text': train_has_symptom_list + train_no_symptom_list, 'label': [1] * len(train_has_symptom_list) + [0] * len(train_no_symptom_list)})
dev_dataset = Dataset.from_dict({'text': dev_has_symptom_list + dev_no_symptom_list, 'label': [1] * len(dev_has_symptom_list) + [0] * len(dev_no_symptom_list)})


def preprocess_function(examples):
    return tokenizer([s.lower() for s in examples["text"]], max_length=MAX_LEN, truncation=True, padding='max_length')


train_dataset = train_dataset.map(preprocess_function, batched=True)
dev_dataset = dev_dataset.map(preprocess_function, batched=True)


training_args = TrainingArguments(
    output_dir="model/" + os.environ["WANDB_NAME"],
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoch_count,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

trainer.save_model("model/" + os.environ["WANDB_NAME"])
