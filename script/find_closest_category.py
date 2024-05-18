import csv
import os
import json
from datetime import date

import jellyfish
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from scipy import spatial
from nltk.tokenize.treebank import TreebankWordDetokenizer


client = OpenAI()


def get_embedding_batched(text_list, model="text-embedding-3-large"):
    response = client.embeddings.create(input=text_list, model=model)

    embedding_list = [r.embedding for r in response.data]

    return embedding_list


if __name__ == "__main__":
    submission_dir = '../data/task1/submission/'
    submission_file = os.path.join(submission_dir, 'test_' + date.today().isoformat() + '.tsv')
    submission_list = []
    os.makedirs(submission_dir, exist_ok=True)
    df = pd.read_pickle('../data/task1/Resource/meddra.pkl')

    # split = "Dev_2024"
    split = "test"
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

    with open("../data/task1/%s/summary.json" % split, 'r') as f:
        summary_dict = json.load(f)

    tweet_dict = dict()
    with open("../data/task1/%s/tweets.tsv" % split, 'r') as f:
        reader = f.readlines()
        for line in reader:
            line = line.strip().split('\t')
            tweet_id = line[0].strip()
            tweet_dict[tweet_id] = line[1].strip()

    span_eval = []

    i = 0
    for line in tqdm(classified_dict):
        tweet_id = line['tweet_id']
        text = line['text']

        gpt_spans = span_dict[tweet_id]
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
                        print('GPT', gpt_spans)
                        print('NER', ner_spans)

                        # raise Exception(f'Check {tweet_id}')

                    from_ = tweet_dict[tweet_id].lower().index(word.lower())
                    to = from_ + len(word)

                    if prev_end is not None and len(tweet_dict[tweet_id][prev_end:from_].strip()):
                        updated_spans.append(span_fix.strip())
                        span_fix = ''

                    span_fix += tweet_dict[tweet_id][from_:to] + ' '

                    prev_end = to

                if len(span_fix):
                    updated_spans.append(span_fix)

        updated_spans = [text.replace("\n", " ").strip() for text in updated_spans]
        updated_spans = [text for text in updated_spans if len(text)]

        spans_with_gpt = []
        for span in updated_spans:
            spans_with_gpt.append('adverse drug event: "' + span + '", context: "' + summary_dict[tweet_id] + '"')

        if not len(updated_spans):
            continue

        embedding_list = get_embedding_batched(spans_with_gpt)
        for emb, span in zip(embedding_list, updated_spans):

            df.loc[:, 'similarity'] = df['embedding'].apply(lambda x: 1 - spatial.distance.cosine(x, emb))
            sorted_df = df.sort_values(['similarity'], ascending=False)
            cat_id = sorted_df['ptid'].iloc[0]

            submission_list.append('\t'.join([tweet_id, span, cat_id]))

    with open(submission_file, 'w') as f:
        f.write('\n'.join(submission_list))

    print(f'Total submissions: {len(submission_list)}')