import csv
import json
import re
from contextlib import suppress
from datasets import load_dataset, ClassLabel, DatasetDict
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk import word_tokenize

from script.classification_dataset_prepare import normalize_string


def build_tweet_dict(data_path):
    result_json = {}

    with open(data_path, 'r') as f:
        reader = f.readlines()

        for line in reader:
            line = line.strip().split('\t')
            if line[0].strip() in result_json:
                raise Exception(f"Duplicate tweet: {line[0].strip()}")

            result_json[line[0].strip()] = line[1].strip()

    return result_json


def prepare_dataset(split):
    if split == 'train':
        spans_data_path = '../data/task1/Train_2024/train_spans_norm_downcast.tsv'
        tweet_data_path = '../data/task1/Train_2024/tweets.tsv'
        json_data_path = '../data/task1/Train_2024/ner.json'
    else:
        spans_data_path = '../data/task1/Dev_2024/norms_downcast.tsv'
        tweet_data_path = '../data/task1/Dev_2024/tweets.tsv'
        json_data_path = '../data/task1/Dev_2024/ner.json'

    tweet_dict = build_tweet_dict(tweet_data_path)
    ner_dict = tweet_dict.copy()

    tokenizer = TreebankWordTokenizer()

    classified_tweets = []
    if 'Dev' in json_data_path:
        with open("../data/task1/Dev_2024/classified.json", 'r') as f:
            classified_dict = json.load(f)
            classified_tweets = list(classified_dict.keys())

    with open(spans_data_path, 'r') as f:
        reader = f.readlines()

    for line in reader:
        line = line.strip().split('\t')
        tweet_id = line[0]
        if tweet_id not in ner_dict:
            raise Exception('Not found')

        span_from = int(line[2])
        span_to = int(line[3])
        span = ner_dict[tweet_id][span_from:span_to]
        shift_left = 0
        shift_right = 0
        space_left = ''
        space_right = ''
        if span_to < len(tweet_dict[tweet_id]) - 1:
            shift_right = 1
            space_right = ' '
        if span_from != 0:
            shift_left = 1
            space_left = ' '

        expected_token_length = len(span)
        token_replacement = ['±'] * len(tokenizer.tokenize(span))
        token_replacement = ' '.join(token_replacement)
        token_replacement += '±' * (expected_token_length - len(token_replacement))

        ner_dict[tweet_id] = ner_dict[tweet_id][:span_from - shift_left] + space_left + token_replacement + space_right + ner_dict[tweet_id][span_to + shift_right:]
        tweet_dict[tweet_id] = tweet_dict[tweet_id][:span_from - shift_left] + space_left + tweet_dict[tweet_id][span_from:span_to] + space_right + tweet_dict[tweet_id][span_to + shift_right:]

    with open(json_data_path.replace('ner', 'ade'), 'r') as f:
        ade_dict = json.load(f)

    with open(json_data_path.replace('ner', 'span'), 'r') as f:
        span_dict = json.load(f)

    with open(json_data_path.replace('ner', 'summary'), 'r') as f:
        summary_dict = json.load(f)

    max_tweet_length = 0
    dataset = []
    ade_dataset = []
    summary_dataset = []
    span_dataset = []
    for tweet_id in ner_dict.keys():
        tokens = tokenizer.tokenize(normalize_string(tweet_dict[tweet_id].strip().lstrip('"').rstrip('"')))
        ner_tags = tokenizer.tokenize(re.sub(r'±+', 'ADE', normalize_string(ner_dict[tweet_id].strip().lstrip('"').rstrip('"'))))

        for i, tag in enumerate(ner_tags):
            if 'ADE' in tag and tag != 'ADE':
                raise Exception(f'Check {tag}, tweet_id: {tweet_id}')
                # split_tokens = [t if len(t) else 'ADE' for t in tag.split('ADE')]
                # split_text = [t for t in split_tokens if t != 'ADE']
                # if len(split_text) != 1:
                #     raise Exception(f'Check text {tag}')
                #
                # tokens[i] = re.sub(split_text[0], split_text[0] + split_text[0], tokens[i])
                # token = [t if len(t) else split_text[0] for t in tokens[i].split(split_text[0])]
                #
                # tokens = tokens[:i] + token + tokens[i + 1:]
                # ner_tags = ner_tags[:i] + split_tokens + ner_tags[i + 1:]

        ner_tags_raw = [tag if tag == 'ADE' else 'O' for tag in ner_tags]
        is_inside = False
        ner_tags = []
        for tag in ner_tags_raw:
            if is_inside:
                if tag == 'ADE':
                    ner_tags.append('I-ADE')
                else:
                    is_inside = False
                    ner_tags.append('O')
            else:
                if tag == 'ADE':
                    ner_tags.append('B-ADE')
                    is_inside = True
                else:
                    ner_tags.append('O')

        if len(ner_tags) != len(tokens):
            raise Exception(f'Check {tweet_id}, {len(tokens)}, {len(ner_tags)}, {tokens}, {" ".join(tokens)} {" ".join(ner_tags)}')
        # dataset.append({
        #     'idx': tweet_id,
        #     'tokens': tokens,
        #     'ner_tags': ner_tags
        # })
        #
        # if tweet_id not in ade_dict:
        #     raise Exception(f'Check ADE {tweet_id}')
        #
        # if tweet_id not in summary_dict:
        #     raise Exception(f'Check Summary {tweet_id}')
        #
        # ade_text = ' [sep] ' + ade_dict[tweet_id]
        # ade_text = normalize_string(ade_text.strip().lstrip('"').rstrip('"'))
        # ade_tokenized = tokenizer.tokenize(ade_text)
        # ade_dataset.append({
        #     'idx': tweet_id,
        #     'tokens': tokens + ade_tokenized,
        #     'ner_tags': ner_tags + ['EXTRA'] * len(ade_tokenized)
        # })
        #
        # summary_text = ' [sep] ' + summary_dict[tweet_id]
        # summary_text = normalize_string(summary_text.strip().lstrip('"').rstrip('"'))
        # summary_tokenized = tokenizer.tokenize(summary_text)
        # summary_dataset.append({
        #     'idx': tweet_id,
        #     'tokens': tokens + summary_tokenized,
        #     'ner_tags': ner_tags + ['EXTRA'] * len(summary_tokenized)
        # })
        #
        # if len(tokens) < 140:
        #     max_tweet_length = max(len(tokens), max_tweet_length)
        #
        # if len(tokens) > 50:
        #     print(tweet_id)

        if 'Dev' in json_data_path:
            if tweet_id not in classified_tweets:
                continue
        elif tweet_id not in span_dict:
            continue

        span_text = ' [sep] ' + ' [sep] '.join(span_dict[tweet_id])
        span_text = normalize_string(span_text.strip().lstrip('"').rstrip('"'))
        span_tokenized = tokenizer.tokenize(span_text)
        span_dataset.append({
            'idx': tweet_id,
            'tokens': tokens + span_tokenized,
            'ner_tags': ner_tags + ['EXTRA'] * len(span_tokenized)
        })

    print(f'Max tweet length: {max_tweet_length}')
    #
    # with open(json_data_path, 'w') as f:
    #     json.dump(dataset, f)
    #
    # with open(json_data_path.replace('ner', 'ade_ner'), 'w') as f:
    #     json.dump(ade_dataset, f)
    #
    # with open(json_data_path.replace('ner', 'summary_ner'), 'w') as f:
    #     json.dump(summary_dataset, f)

    with open(json_data_path.replace('ner', 'span_ner'), 'w') as f:
        json.dump(span_dataset, f)

    tag_mapping = {
        'O': 0,
        'B-ADE': 1,
        'I-ADE': 2,
        'EXTRA': -100
    }

    # dataset = load_dataset('json', data_files=json_data_path, split="train")
    # new_features = dataset.features.copy()
    # new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    # dataset = dataset.cast(new_features)
    #
    # ade_dataset = load_dataset('json', data_files=json_data_path.replace('ner', 'ade_ner'), split="train")
    # new_features = ade_dataset.features.copy()
    # new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    # ade_dataset = ade_dataset.cast(new_features)
    #
    # summary_dataset = load_dataset('json', data_files=json_data_path.replace('ner', 'summary_ner'), split="train")
    # new_features = summary_dataset.features.copy()
    # new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    # summary_dataset = summary_dataset.cast(new_features)

    span_dataset = load_dataset('json', data_files=json_data_path.replace('ner', 'span_ner'), split="train")
    new_features = span_dataset.features.copy()
    new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
    span_dataset = span_dataset.cast(new_features)

    return dataset, ade_dataset, summary_dataset, span_dataset


if __name__ == '__main__':
    dev_dataset, dev_ade_dataset, dev_summary_dataset, dev_span_dataset = prepare_dataset('dev')
    train_dataset, train_ade_dataset, train_summary_dataset, train_span_dataset = prepare_dataset('train')

    dataset_path = '../data/task1/2024_ner_gpt4'

    # dataset = DatasetDict({'train': train_dataset, 'dev': dev_dataset})
    # dataset.save_to_disk(dataset_path)
    #
    # ade_dataset = DatasetDict({'train': train_ade_dataset, 'dev': dev_ade_dataset})
    # ade_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_ade_ner_gpt4'))
    #
    # summary_dataset = DatasetDict({'train': train_summary_dataset, 'dev': dev_summary_dataset})
    # summary_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_summary_ner_gpt4'))

    span_dataset = DatasetDict({'train': train_span_dataset, 'dev': dev_span_dataset})
    span_dataset.save_to_disk(dataset_path.replace('2024_ner_gpt4', '2024_span_ner_gpt4_classified'))
