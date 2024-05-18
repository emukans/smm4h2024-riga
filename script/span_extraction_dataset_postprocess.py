import csv
import json
import os
from glob import glob

from script.classification_dataset_prepare import normalize_string


def build_tweet_dict(data_path):
    result_json = {}

    with open(data_path, 'r') as f:
        reader = f.readlines()

        for line in reader:
            line = line.strip().split('\t')
            if line[0].strip() in result_json:
                raise Exception(f"Duplicate tweet: {line[0].strip()}")

            result_json[line[0].strip()] = normalize_string(line[1].strip())

    return result_json


if __name__ == '__main__':
    train_span_data_path = '../data/task1/Train_2024/tweet_span_classification_prefix'
    train_dataset_path = '../data/task1/Train_2024/span_extraction/response'
    train_positive_data_path = '../data/task1/Train_2024/train_spans_norm_downcast.tsv'
    train_tweet_data_path = '../data/task1/Train_2024/tweets.tsv'

    dev_span_data_path = '../data/task1/Dev_2024/tweet_span_classification_prefix'
    dev_dataset_path = '../data/task1/Dev_2024/span_extraction/response'
    dev_positive_data_path = '../data/task1/Dev_2024/norms_downcast.tsv'
    dev_tweet_data_path = '../data/task1/Dev_2024/tweets.tsv'

    span_data_path = train_span_data_path
    dataset_path = train_dataset_path
    positive_data_path = train_positive_data_path
    tweet_data_path = train_tweet_data_path

    extracted_span_list = glob(os.path.join(dataset_path, '*.txt'))
    extracted_span_list = [os.path.splitext(os.path.basename(name))[0] for name in extracted_span_list]

    with open(positive_data_path, 'r') as f:
        reader = f.readlines()

        has_symptom_list = []
        for line in reader:
            line = line.strip().split('\t')
            has_symptom_list.append(line[0])

        has_symptom_list = set(has_symptom_list)

    tweet_positive_list = []
    tweet_negative_list = []

    total_entries = 0
    has_span_correctly_classified = 0
    has_span_incorrectly_classified = 0
    no_span_correctly_classified = 0
    no_span_incorrectly_classified = 0

    tweet_json = dict()
    with open(tweet_data_path, 'r') as f:
        reader = f.readlines()

        for line in reader:
            total_entries += 1

            line = line.strip().split('\t')
            tweet_id = line[0]
            tweet_text = normalize_string(line[1].lower().strip())
            has_extracted_span = False
            with open(os.path.join(dataset_path, f'{tweet_id}.txt'), 'r') as f:
                response = f.read().splitlines()
                for r in response:
                    span = r.lower().lstrip('span:').strip()
                    if 'null' in span and span != 'null':
                        raise Exception(f"Tweet {tweet_id}")

                    if span == 'null' or not len(span):
                        continue

                    has_extracted_span = True
                    tweet_text = f'[{span}]' + tweet_text
                    # append_to_end = False
                    # try:
                    #     index_from = tweet_text.index(span)
                    # except ValueError:
                    #     append_to_end = True
                    #
                    # if append_to_end:
                    #     tweet_text += f'<span>{span}</span>'
                    # else:
                    #     tweet_text = f'{tweet_text[:index_from]}<span>{tweet_text[index_from:index_from + len(span)]}</span>{tweet_text[index_from + len(span):]}'

            tweet_json[tweet_id] = tweet_text
            if line[0] in has_symptom_list:
                if has_extracted_span:
                    has_span_correctly_classified += 1
                else:
                    has_span_incorrectly_classified += 1
                tweet_positive_list.append(tweet_text)
            else:
                if has_extracted_span:
                    no_span_incorrectly_classified += 1
                else:
                    no_span_correctly_classified += 1
                tweet_negative_list.append(tweet_text)

    if sum([has_span_incorrectly_classified, has_span_correctly_classified, no_span_correctly_classified, no_span_incorrectly_classified]) != total_entries:
        raise Exception(f"Calculation is wrong")

    accuracy = (has_span_correctly_classified + no_span_correctly_classified) / total_entries
    precision = has_span_correctly_classified / (has_span_correctly_classified + no_span_incorrectly_classified)
    recall = has_span_correctly_classified / (has_span_correctly_classified + has_span_incorrectly_classified)
    f1 = 2 * precision * recall / (precision + recall)
    print(f'Total entries: {total_entries}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1: {f1:.2f}')

    os.makedirs(span_data_path, exist_ok=True)

    with open(os.path.join(span_data_path, 'has_symptom.txt'), 'w') as f:
        f.write('\n'.join(set(tweet_positive_list)))

    with open(os.path.join(span_data_path, 'no_symptom.txt'), 'w') as f:
        f.write('\n'.join(set(tweet_negative_list)))

    with open(os.path.join(span_data_path, 'tweet.json'), 'w') as f:
        json.dump(tweet_json, f)
