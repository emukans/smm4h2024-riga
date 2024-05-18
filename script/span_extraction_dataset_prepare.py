import csv
import json
import os

from script.classification_dataset_prepare import normalize_string


def build_tweet_dict(data_path):
    result_json = {}

    with open(data_path, 'r') as f:
        reader = f.readlines()

        for line in reader:
            line = line.strip().split('\t')
            if line[0].strip() in result_json:
                raise Exception(f"Duplicate tweet: {line[0].strip()}")

            result_json[line[0].strip()] = normalize_string(line[1].strip().lstrip('"').rstrip('"'))

    return result_json


if __name__ == '__main__':
    train_tweets_data_path = '../data/task1/Train_2024/tweets.tsv'
    train_dataset_path = '../data/task1/Train_2024/span_extraction'

    dev_tweets_data_path = '../data/task1/Dev_2024/tweets.tsv'
    dev_dataset_path = '../data/task1/Dev_2024/span_extraction'

    os.makedirs(train_dataset_path, exist_ok=True)
    train_json = build_tweet_dict(train_tweets_data_path)

    os.makedirs(dev_dataset_path, exist_ok=True)
    dev_json = build_tweet_dict(dev_tweets_data_path)

    with open(os.path.join(train_dataset_path, 'tweets.json'), 'w') as f:
        json.dump(train_json, f)

    with open(os.path.join(dev_dataset_path, 'tweets.json'), 'w') as f:
        json.dump(dev_json, f)
