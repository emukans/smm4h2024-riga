from random import shuffle, seed

import torch
import wandb
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, EarlyStoppingCallback

import os

import numpy as np

from datasets import Dataset, load_from_disk, load_metric
import evaluate
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification


seed(42)
np.random.seed(42)


# checkpoint = "distilbert/distilbert-base-uncased"
# checkpoint = "cardiffnlp/twitter-roberta-base-ner7-latest"
checkpoint = "Clinical-AI-Apollo/Medical-NER"
# checkpoint = "dslim/bert-large-NER"
# checkpoint = "d4data/biomedical-ner-all"


MAX_LEN = 70
# batch_size = 2
batch_size = 64
epoch_count = 5000
learning_rate = 2e-5
downsample = 1
save_total_limit = 5
print_example = False


# os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_PROJECT"] = "smm4h2024-task1-tweet-ner"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_NAME"] = f"{checkpoint}/tweet_span_ade_ner_classified/gpt4-lr-{learning_rate}-downsample-{downsample}-max_len-{MAX_LEN}-ignore-bio"
# os.environ["WANDB_NOTES"] = "Spans extracted by GPT3.5 from tweets, classification. Downample 0.2"

# dataset = load_from_disk('data/2024_ade_ner_gpt4_bio')
# dataset = load_from_disk('data/2024_summary_ner_gpt4_bio')
# dataset = load_from_disk('data/2024_ner_gpt4')
# dataset = load_from_disk('data/2024_span_ner_gpt4')
dataset = load_from_disk('data/2024_span_ner_gpt4_classified')

# dataset['train'] = dataset['train'].select([0, 1, 2, 3, 4, 5])
# dataset['dev'] = dataset['dev'].select([0, 1, 2, 3, 4, 5])


id2label = {0: "O", 1: "B-ADE", 2: "I-ADE"}
label2id = {"O": 0, "B-ADE": 1, "I-ADE": 2}
# id2label = {0: "O", 1: "B-ADE", 2: "I-ADE", -100: "EXTRA"}
# label2id = {"O": 0, "B-ADE": 1, "I-ADE": 2, "EXTRA": -100}
label_list = dataset["train"].features[f"ner_tags"].feature.names
tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(
    checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


train_dataset = dataset['train']
dev_dataset = dataset['dev']

log_params = {
    'train_size': len(dataset['train']),
    'dev_size': len(dataset['dev']),
    'model_size': model.num_parameters(),
    'max_len': MAX_LEN,
}

if downsample != 1:
    has_ade_ids = []
    for i, item in enumerate(train_dataset):
        if 1 in item['ner_tags']:
            has_ade_ids.append(item['idx'])

    no_ade_ids = list(set(train_dataset['idx']) - set(has_ade_ids))
    no_ade_ids = no_ade_ids[:round(len(no_ade_ids) * downsample)]
    ids_to_keep = no_ade_ids + has_ade_ids
    train_dataset = train_dataset.filter(lambda item: item['idx'] in ids_to_keep)

    log_params['train_size'] = len(train_dataset)
    log_params['positive_sample_proportion_train'] = len(has_ade_ids) / len(train_dataset)
    log_params['downsample'] = downsample

    has_ade_ids = []
    for i, item in enumerate(dev_dataset):
        if 1 in item['ner_tags']:
            has_ade_ids.append(item['idx'])
    log_params['positive_sample_proportion_dev'] = len(has_ade_ids) / len(dev_dataset)


wandb.init()
wandb.log(log_params)


def tokenize_and_align_labels(examples):
    """
    After tokenization, a word is split into multiple tokens. This function assigns the same POS tag for every token of the word.
    """
    global print_example

    tokenized_inputs = tokenizer(examples["tokens"], max_length=MAX_LEN, truncation=True, padding='max_length', is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                # 3 is EXTRA token
                if label[word_idx] == 3:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if label[word_idx] == 3:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    if print_example:
        print_example = False
        print('Before: ', examples['ner_tags'][:5])
        print('Before: ', examples['tokens'][:5])
        print('After: ', tokenized_inputs['labels'][:5])
        print('After: ', tokenized_inputs['input_ids'][:5])
        exit()

    return tokenized_inputs


train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
dev_dataset = dev_dataset.map(tokenize_and_align_labels, batched=True)


training_args = TrainingArguments(
    output_dir="model/" + os.environ["WANDB_NAME"],
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoch_count,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=save_total_limit,
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
