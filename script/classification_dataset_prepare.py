import csv
import json
import os
import re


def normalize_string(s):
    s = s.strip().lstrip('"').rstrip('"')
    s = re.sub(r'@user_*', '[user]', s, flags=re.I)
    s = re.sub(r'httpurl_*', '[url]', s, flags=re.I)
    s = re.sub(r'(\[user]\s)+', r'[user] ', s)
    s = re.sub(r'(\[url]\s)+', r'[url] ', s)

    return s


if __name__ == '__main__':
    train_positive_data_path = '../data/task1/Train_2024/train_spans_norm_downcast.tsv'
    train_tweets_data_path = '../data/task1/Train_2024/tweets.tsv'
    train_dataset_path = '../data/task1/Train_2024/classification'

    dev_positive_data_path = '../data/task1/Dev_2024/norms_downcast.tsv'
    dev_tweets_data_path = '../data/task1/Dev_2024/tweets.tsv'
    dev_dataset_path = '../data/task1/Dev_2024/classification'

    # test_dataset_path = '../data/task1/test/tweets.tsv'
    #
    # test_data_dict = dict()
    # with open(test_dataset_path, 'r') as f:
    #     reader = f.readlines()
    #
    #     for line in reader:
    #         line = line.strip().split('\t')
    #
    #         test_data_dict[line[0].strip()] = normalize_string(line[1].strip().lstrip('"').rstrip('"'))
    #
    # with open(test_dataset_path.replace('.tsv', '.json'), 'w') as f:
    #     json.dump(test_data_dict, f)

    with open(train_positive_data_path, 'r') as f:
        reader = f.readlines()

        train_has_symptom_list = []
        for line in reader:
            line = line.strip().split('\t')
            train_has_symptom_list.append(line[0])

        train_has_symptom_list = set(train_has_symptom_list)

    dev_span_json = dict()
    with open(dev_positive_data_path, 'r') as f:
        reader = f.readlines()

        dev_has_symptom_list = []
        for line in reader:
            line = line.strip().split('\t')
            dev_has_symptom_list.append(line[0])

            tweet_id = line[0].strip()
            if tweet_id not in dev_span_json:
                dev_span_json[tweet_id] = []

            dev_span_json[tweet_id].append(line[-2])

        dev_has_symptom_list = set(dev_has_symptom_list)

    with open(dev_positive_data_path.replace('norms_downcast.tsv', 'gt_span.json'), 'w') as f:
        json.dump(dev_span_json, f)

    os.makedirs(train_dataset_path, exist_ok=True)

    with open(train_tweets_data_path, 'r') as f:
        reader = f.readlines()

        train_tweet_positive_list = []
        train_tweet_negative_list = []
        for line in reader:
            line = line.strip().split('\t')
            if line[0] in train_has_symptom_list:
                train_tweet_positive_list.append(normalize_string(line[1]))
            else:
                train_tweet_negative_list.append(normalize_string(line[1]))

        with open(os.path.join(train_dataset_path, 'has_symptom.txt'), 'w') as f:
            f.write('\n'.join(set(train_tweet_positive_list)))

        with open(os.path.join(train_dataset_path, 'no_symptom.txt'), 'w') as f:
            f.write('\n'.join(set(train_tweet_negative_list)))

    os.makedirs(dev_dataset_path, exist_ok=True)

    with open(dev_tweets_data_path, 'r') as f:
        reader = f.readlines()

        dev_tweet_positive_list = []
        dev_tweet_negative_list = []
        for line in reader:
            line = line.strip().split('\t')
            if line[0] in dev_has_symptom_list:
                dev_tweet_positive_list.append(normalize_string(line[1]))
            else:
                dev_tweet_negative_list.append(normalize_string(line[1]))

        with open(os.path.join(dev_dataset_path, 'has_symptom.txt'), 'w') as f:
            f.write('\n'.join(set(dev_tweet_positive_list)))

        with open(os.path.join(dev_dataset_path, 'no_symptom.txt'), 'w') as f:
            f.write('\n'.join(set(dev_tweet_negative_list)))
