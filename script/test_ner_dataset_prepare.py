import csv
import json
import os

from script.classification_dataset_prepare import normalize_string


def extract_summary(tweet_data_path, summary_data_path):

    result_json = {}
    summary_json = {}
    ade_json = {}
    span_json = {}

    with open(tweet_data_path, 'r') as f:
        reader = f.readlines()

        for line in reader:
            line = line.strip().split('\t')
            tweet_id = line[0].strip()
            text = line[1].strip().lstrip('"').rstrip('"')
            file_location = os.path.join(summary_data_path, f'{tweet_id}.txt')

            if not os.path.exists(file_location):
                if 'dev' in tweet_data_path:
                    raise Exception(f'File not exists: {file_location}')
                else:
                    continue

            span_list = []
            with open(file_location, 'r') as r:
                span_reader = r.readlines()
                for line in span_reader:
                    if 'SPAN:' not in line:
                        raise Exception(f'Check {tweet_id}')
                    span = line.lstrip('SPAN:').strip()
                    if span.lower() != 'null':
                        span_list.append(span)

            span_json[tweet_id] = span_list

            # summary = ''
            # has_summary = False
            # ade = ''
            # if not os.path.exists(file_location):
            #     raise Exception(f'File not exists: {file_location}')
            #
            # with open(file_location, 'r') as r:
            #     summary_list = r.readlines()
            #     for summary_line in summary_list:
            #         if 'Summary:' in summary_line:
            #             summary = summary_line.lstrip('Summary:').strip()
            #             has_summary = True
            #         if 'ADE:' in summary_line:
            #             ade = summary_line.lstrip('ADE:').strip()
            #             if ade.lower() == 'null':
            #                 ade = ''
            #
            # if not has_summary:
            #     raise Exception(tweet_id)
            #
            # summary_json[tweet_id] = normalize_string(summary.strip().lstrip('"').rstrip('"'))
            # ade_json[tweet_id] = normalize_string(ade.strip().lstrip('"').rstrip('"'))

            if tweet_id in result_json:
                raise Exception(f"Duplicate tweet: {tweet_id}")

            result_json[tweet_id] = normalize_string(text)

    return result_json, summary_json, ade_json, span_json


def save_classified_data(has_symptom_list, tweet_dict, data_path):
    positive_list = []
    negative_list = []
    for tweet_id, text in tweet_dict.items():
        if tweet_id in has_symptom_list:
            positive_list.append(normalize_string(text.strip().lstrip('"').rstrip('"')))
        else:
            negative_list.append(normalize_string(text.strip().lstrip('"').rstrip('"')))

    os.makedirs(data_path, exist_ok=True)

    with open(os.path.join(data_path, 'has_symptom.txt'), 'w') as f:
        f.write('\n'.join(set(positive_list)))

    with open(os.path.join(data_path, 'no_symptom.txt'), 'w') as f:
        f.write('\n'.join(set(negative_list)))


if __name__ == '__main__':
    train_positive_data_path = '../data/task1/Train_2024/train_spans_norm_downcast.tsv'
    train_tweets_data_path = '../data/task1/Train_2024/tweets.tsv'
    train_summary_data_path = '../data/task1/Train_2024/ade_extraction_gpt4/response'
    train_dataset_path = '../data/task1/Train_2024'

    dev_positive_data_path = '../data/task1/Dev_2024/norms_downcast.tsv'
    dev_tweets_data_path = '../data/task1/Dev_2024/tweets.tsv'
    dev_summary_data_path = '../data/task1/Dev_2024/ade_extraction_gpt4/response'
    dev_dataset_path = '../data/task1/Dev_2024'

    test_tweets_data_path = '../data/task1/test/tweets.tsv'
    test_summary_data_path = '../data/task1/test/ade_extraction_gpt4/response'
    test_dataset_path = '../data/task1/test'

    os.makedirs(test_dataset_path, exist_ok=True)
    test_json, test_summary, test_ade, test_span_json = extract_summary(test_tweets_data_path, test_summary_data_path)

    with open(os.path.join(test_dataset_path, 'tweets.json'), 'w') as f:
        json.dump(test_json, f)

    test_span_merged = {}
    for tweet_id, text in test_json.items():
        span_text = ' [sep] ' + ' [sep] '.join(test_span_json[tweet_id])
        span_text = normalize_string(span_text.strip().lstrip('"').rstrip('"'))
        test_span_merged[tweet_id] = normalize_string(
            text.strip().lstrip('"').rstrip('"')) + span_text


    with open(os.path.join(test_dataset_path, 'span.json'), 'w') as f:
        json.dump(test_span_json, f)

    with open(os.path.join(test_dataset_path, 'span_merged.json'), 'w') as f:
        json.dump(test_span_merged, f)

    print(len(test_span_merged))
    # with open(os.path.join(test_dataset_path, 'summary_merged.json'), 'w') as f:
    #     json.dump(test_summary_merged, f)

    exit()

    os.makedirs(train_dataset_path, exist_ok=True)
    train_json, train_summary, train_ade, train_span_json = extract_summary(train_tweets_data_path, train_summary_data_path)

    os.makedirs(dev_dataset_path, exist_ok=True)
    dev_json, dev_summary, dev_ade, dev_span_json = extract_summary(dev_tweets_data_path, dev_summary_data_path)

    with open(os.path.join(train_dataset_path, 'tweets.json'), 'w') as f:
        json.dump(train_json, f)

    with open(os.path.join(train_dataset_path, 'summary.json'), 'w') as f:
        json.dump(train_summary, f)

    with open(os.path.join(train_dataset_path, 'ade.json'), 'w') as f:
        json.dump(train_ade, f)

    with open(os.path.join(train_dataset_path, 'span.json'), 'w') as f:
        json.dump(train_span_json, f)

    with open(os.path.join(dev_dataset_path, 'tweets.json'), 'w') as f:
        json.dump(dev_json, f)

    with open(os.path.join(dev_dataset_path, 'summary.json'), 'w') as f:
        json.dump(dev_summary, f)

    with open(os.path.join(dev_dataset_path, 'ade.json'), 'w') as f:
        json.dump(dev_ade, f)

    with open(os.path.join(dev_dataset_path, 'span.json'), 'w') as f:
        json.dump(dev_span_json, f)

    with open(train_positive_data_path, 'r') as f:
        reader = f.readlines()

        train_has_symptom_list = []
        for line in reader:
            line = line.strip().split('\t')
            train_has_symptom_list.append(line[0])

        train_has_symptom_list = set(train_has_symptom_list)

    with open(dev_positive_data_path, 'r') as f:
        reader = f.readlines()

        dev_has_symptom_list = []
        for line in reader:
            line = line.strip().split('\t')
            dev_has_symptom_list.append(line[0])

        dev_has_symptom_list = set(dev_has_symptom_list)

    save_classified_data(train_has_symptom_list, train_summary, os.path.join(train_dataset_path, 'summary_classification'))
    save_classified_data(train_has_symptom_list, train_json, os.path.join(train_dataset_path, 'tweet_classification'))
    save_classified_data(train_has_symptom_list, train_ade, os.path.join(train_dataset_path, 'ade_classification'))
    save_classified_data(dev_has_symptom_list, dev_summary, os.path.join(dev_dataset_path, 'summary_classification'))
    save_classified_data(dev_has_symptom_list, dev_json, os.path.join(dev_dataset_path, 'tweet_classification'))
    save_classified_data(dev_has_symptom_list, dev_ade, os.path.join(dev_dataset_path, 'ade_classification'))

    train_ade_merged = {}
    for tweet_id, text in train_json.items():
        train_ade_merged[tweet_id] = normalize_string(
            text.strip().lstrip('"').rstrip('"')) + ' <sep> ' + normalize_string(
            train_ade[tweet_id].strip().lstrip('"').rstrip('"'))

    dev_ade_merged = {}
    for tweet_id, text in dev_json.items():
        dev_ade_merged[tweet_id] = normalize_string(
            text.strip().lstrip('"').rstrip('"')) + ' <sep> ' + normalize_string(
            dev_ade[tweet_id].strip().lstrip('"').rstrip('"'))

    train_summary_merged = {}
    for tweet_id, text in train_json.items():
        train_summary_merged[tweet_id] = normalize_string(
            text.strip().lstrip('"').rstrip('"')) + ' <sep> ' + normalize_string(
            train_summary[tweet_id].strip().lstrip('"').rstrip('"'))

    dev_summary_merged = {}
    for tweet_id, text in dev_json.items():
        dev_summary_merged[tweet_id] = normalize_string(
            text.strip().lstrip('"').rstrip('"')) + ' <sep> ' + normalize_string(
            dev_summary[tweet_id].strip().lstrip('"').rstrip('"'))

    save_classified_data(train_has_symptom_list, train_ade_merged, os.path.join(train_dataset_path, 'ade_merged_classification'))
    save_classified_data(train_has_symptom_list, train_summary_merged, os.path.join(train_dataset_path, 'summary_merged_classification'))
    save_classified_data(dev_has_symptom_list, dev_ade_merged, os.path.join(dev_dataset_path, 'ade_merged_classification'))
    save_classified_data(dev_has_symptom_list, dev_summary_merged, os.path.join(dev_dataset_path, 'summary_merged_classification'))

    with open(os.path.join(train_dataset_path, 'ade_merged.json'), 'w') as f:
        json.dump(train_ade_merged, f)

    with open(os.path.join(dev_dataset_path, 'ade_merged.json'), 'w') as f:
        json.dump(dev_ade_merged, f)

    with open(os.path.join(train_dataset_path, 'summary_merged.json'), 'w') as f:
        json.dump(train_summary_merged, f)

    with open(os.path.join(dev_dataset_path, 'summary_merged.json'), 'w') as f:
        json.dump(dev_summary_merged, f)
