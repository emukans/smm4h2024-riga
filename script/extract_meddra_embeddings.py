import itertools
import os

from openai import OpenAI
from tqdm import tqdm

from script.data_preprocess import get_meddra_dict
import pandas as pd


client = OpenAI()


def get_embedding_batched(text_list, model="text-embedding-3-large"):
    text_list = [text.replace("\n", " ") for text in text_list]

    response = client.embeddings.create(input=text_list, model=model)

    embedding_list = [r.embedding for r in response.data]

    return embedding_list


def chunked(it, size):
    it = iter(it)
    while True:
        p = tuple(itertools.islice(it, size))
        if not p:
            break
        yield p


if __name__ == '__main__':
    meddra_path = '../data/task1/MedDRA/llt.asc'
    pt_dict, llt_dict = get_meddra_dict(meddra_path)

    df = []
    for item_list in tqdm(chunked(pt_dict.items(), 100)):
        text_list = [i.text for _, i in item_list]
        embedding_list = get_embedding_batched(text_list)

        for (_, item), embedding in zip(item_list, embedding_list):
            df.append({
                'text': item.text,
                'ptid': item.ptid,
                'embedding': embedding
            })

    df = pd.DataFrame(df)
    print(len(df))
    print(df.head())
    df.to_pickle('../data/task1/meddra.pkl')
    # df = pd.read_pickle('../data/task1/meddra.pkl')
