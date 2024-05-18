import csv
import os
import json
from datetime import date

from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from scipy import spatial
from nltk.tokenize.treebank import TreebankWordDetokenizer


if __name__ == "__main__":
    split = 'test'
    # split = 'Dev_2024'
    if split != 'test':
        with open("../data/task1/%s/classified.json" % split, 'r') as f:
            classified_dict = json.load(f)
    else:
        with open("../data/task1/%s/classified.csv" % split, 'r') as f:
            reader = csv.DictReader(f)
            classified_dict = []
            for line in reader:
                if not int(line['predicted']):
                    continue
                classified_dict.append(line)

    with open("../data/task1/%s/ner_prediction.json" % split, 'r') as f:
        ner_dict = json.load(f)

    with open("../data/task1/%s/span.json" % split, 'r') as f:
        span_dict = json.load(f)

    gt_dict = {}
    if split != 'test':
        with open("../data/task1/%s/gt_span.json" % split, 'r') as f:
            gt_dict = json.load(f)

    span_extraction_dir = '../data/task1/%s/span_extraction/response' % split

    tweet_dict = dict()
    with open("../data/task1/%s/tweets.tsv" % split, 'r') as f:
        reader = f.readlines()
        for line in reader:
            line = line.strip().split('\t')
            tweet_id = line[0].strip()
            tweet_dict[tweet_id] = line[1].strip()

    span_eval = []

    for line in tqdm(classified_dict):
        tweet_id = line['tweet_id']
        text = line['text']

        ner_spans = ner_dict[tweet_id]

        updated_spans = []
        for span in ner_spans:
            if span in tweet_dict[tweet_id].lower():
                from_ = tweet_dict[tweet_id].lower().index(span.lower())
                to = from_ + len(span)
                updated_spans.append(tweet_dict[tweet_id][from_:to])
            else:
                span_fix = ''
                prev_end = None
                for word in span.split():
                    if word.lower() not in tweet_dict[tweet_id].lower():
                        print('Text', tweet_dict[tweet_id])
                        print('GPT', span_dict[tweet_id])
                        print('NER', ner_spans)

                        raise Exception(f'Check {tweet_id}')

                    from_ = tweet_dict[tweet_id].lower().index(word.lower())
                    to = from_ + len(word)

                    if prev_end is not None and len(tweet_dict[tweet_id][prev_end:from_].strip()):
                        updated_spans.append(span_fix.strip())
                        span_fix = ''

                    span_fix += tweet_dict[tweet_id][from_:to] + ' '

                    prev_end = to

                if len(span_fix):
                    updated_spans.append(span_fix)

        gt_span = gt_dict[tweet_id] if tweet_id in gt_dict else []
        span_eval.append({
            'tweet_id': tweet_id,
            'text': tweet_dict[tweet_id],
            'ner': '; '.join(updated_spans),
            'gpt4': '; '.join(span_dict[tweet_id]),
        })

    with open('../data/task1/%s/span_prediction.csv' % split, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['tweet_id', 'text', 'ner', 'gpt4'])
        writer.writeheader()
        for line in span_eval:
            writer.writerow(line)
