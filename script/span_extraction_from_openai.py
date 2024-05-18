import csv
import json
import os
from glob import glob
import sys

import openai
from tqdm import tqdm


# GPT-4-turbo
# TASK: Extract specific spans from tweets that describe adverse drug events (ADEs), focusing on the negative effects or reactions caused by medications. The extracted spans should precisely capture the essence of the adverse event mentioned in the tweet.
#
# RESTRICTIONS:
# 1. Span Length: Each extracted span must be no more than 5 words long. This limit is to ensure the extracted information is concise and directly related to the adverse event.
# 2. Multiple Span Selection: You are allowed to select more than one span from a single tweet if multiple adverse events or significant details about a single event are mentioned. Each span should be separately identified.
# 3. No Span: If the tweet does not mention any adverse drug events or related negative effects, the output should be "null". This indicates that no relevant adverse event information was found in the tweet.
# 4. Span Accuracy: The spans must accurately reflect the adverse event or negative effect mentioned, without altering the meaning or leaving out critical context. The span should be a direct quote from the tweet without paraphrasing or summarization.
#
# EXAMPLES:
# - INPUT: "ciprofloxacin: how do you expect to sleep when your stomach is a cement mixer?"
#   - SPAN: "sleep"
#   - SPAN: "stomach is a cement mixer"
# - INPUT: "[user] yes ma'am. MTX of 20mg weekly too. Tried Enbrel before this, and will do Remicade if this doesn't help. #rheum"
#   - SPAN: "null"
# - INPUT: "So glad I'm off #effexor, so sad it ruined my teeth. #tip Please be carefull w/ taking #antideppresiva and read about it 1st! #venlafaxine"
#   - SPAN: "ruined my teeth"
# - INPUT: "Bananas contain a natural chemical which can make a person happy. This same chemical is also found in Prozac."
#   - SPAN: "null"
# - INPUT: "Iwwwwww RT\"[user]: At humira's house\""
#   - SPAN: "null"
#
# Ensure that the spans you extract are directly related to the adverse effects or negative reactions caused by the drugs mentioned, adhering to the specified restrictions for accurate and relevant span extraction.

# GPT-4
# TASK: The task involves extracting relevant spans of text from tweets that indicate adverse drug events. The span of text should be directly related to the negative effects or adverse reactions caused by the drug mentioned in the tweet. The length of the span should not exceed 5 words. Multiple spans can be selected from a single tweet if they indicate different adverse events. If no relevant span is found in the tweet, the output should be 'null'.
#
# RESTRICTIONS: The span of text extracted should not exceed 5 words in length. The span should directly indicate an adverse drug event or negative effect caused by the drug. Multiple spans can be selected from a single tweet, but each span should indicate a different adverse event. If no relevant span is found in the tweet, the output should be 'null'. The span should not include any irrelevant words or phrases that do not contribute to the understanding of the adverse event.


# GPT-3.5
# TASK: Extract spans from tweets related to adverse drug events.
#
# RESTRICTIONS:
# 1. Span length should be 5 words or fewer.
# 2. Multiple span selections are allowed.
# 3. Place different drug events types into a separate spans.
# 4. If no span related to adverse drug events is present in the tweet, mark it as "null."
#
# ---
# INPUT: ciprofloxacin: how do you expect to sleep when your stomach is a cement mixer?
# OUTPUT:
# SPAN: sleep
# SPAN: stomach is a cement mixer
#
# ---
# INPUT: [user] yes ma'am. MTX of 20mg weekly too. Tried Enbrel before this, and will do Remicade if this doesn't help. #rheum
# OUTPUT:
# SPAN: null
#
# ---
# INPUT: So glad I'm off #effexor, so sad it ruined my teeth. #tip Please be careful w/ taking #antidepressants and read about it 1st! #venlafaxine
# OUTPUT:
# SPAN: ruined my teeth
#
# ---
# INPUT: Bananas contain a natural chemical which can make a person happy. This same chemical is also found in Prozac.
# OUTPUT:
# SPAN: null
#
# ---
# INPUT: "trying to distract with masterchef, twitter, texting &amp; candy crush but i'm anxious &amp; have stomach &amp; head pain #venlafaxine #withdrawal"
# OUTPUT:


prompt_span = """TASK: Extract spans from tweets related to adverse drug events.

RESTRICTIONS:
1. Each span length should be 5 words or less.
2. Multiple span selections are allowed.
3. Place different drug events types into a separate spans.
4. If no span related to adverse drug events is present in the tweet, mark it as "null".

---
INPUT: "trying to distract with masterchef, twitter, texting &amp; candy crush but i'm anxious &amp; have stomach &amp; head pain #venlafaxine #withdrawal"
OUTPUT:
SPAN: anxious
SPAN: stomach pain
SPAN: head pain
SPAN: withdrawal

---
INPUT: [user] yes ma'am. MTX of 20mg weekly too. Tried Enbrel before this, and will do Remicade if this doesn't help. #rheum
OUTPUT:
SPAN: null

---
INPUT: So glad I'm off #effexor, so sad it ruined my teeth. #tip Please be careful w/ taking #antidepressants and read about it 1st! #venlafaxine
OUTPUT:
SPAN: ruined my teeth

---
INPUT: Bananas contain a natural chemical which can make a person happy. This same chemical is also found in Prozac.
OUTPUT:
SPAN: null

---
INPUT: "[user] i found the humira to fix all my crohn's issues, but cause other issues. i went off it due to issues w nerves/muscle spasms"
OUTPUT:
SPAN: nerves
SPAN: muscle spasms

---
INPUT: {tweet}
OUTPUT:
"""

prompt_summary = '''
You will be provided with a tweet. Summarise it into a brief sentence and highlight already happened adverse drug events (ADE) if there are any related to drugs.
Format:
Summary: text
ADE: text or null
---
Tweet:
"""
{tweet}
"""
'''

initial_ade_mining_prompt = '''
You will be provided with a tweet. Highlight already happened adverse drug events (ADE) if there are any related to drugs. Omit the ADE context, output just ADE. Write the span exactly as in the tweet. If there are many different ADEs by context within the same span, then split it. If ADE repeats, then put each mention on a new line. If there are multiple spans then put each on a new line.
---
Format:
SPAN: text or null
---
Samples:
Tweet:
"""
[user] if #avelox has hurt your liver, avoid tylenol always, as it further damages liver, eat grapefruit unless taking cardiac drugs
"""
SPAN: hurt your liver
---
Tweet:
"""
losing it. could not remember the word power strip. wonder which drug is doing this memory lapse thing. my guess the cymbalta. #helps
"""
SPAN: not remember
SPAN: memory lapse
Tweet:
"""
is adderall a performance enhancing drug for mathletes?
"""
SPAN: null
---
Tweet:
"""
[user] i found the humira to fix all my crohn's issues, but cause other issues. i went off it due to issues w nerves/muscle spasms
"""
'''

ade_mining = '''
You will be provided with a tweet. Your task is to identify and highlight any adverse drug events (ADEs) mentioned in relation to drug use. Only the exact phrases describing the ADEs should be outputted, without including any additional context. Each ADE should be listed on a new line. If the same ADE is mentioned multiple times, each occurrence should be listed separately. If multiple different ADEs are identified within the same tweet, they should be listed on separate lines. If no ADEs are found, output "null".
---
Format:
SPAN: text or null
---
Samples:
Tweet:
"""
[user] if #avelox has hurt your liver, avoid tylenol always, as it further damages liver, eat grapefruit unless taking cardiac drugs
"""
SPAN: hurt your liver
---
Tweet:
"""
losing it. could not remember the word power strip. wonder which drug is doing this memory lapse thing. my guess the cymbalta. #helps
"""
SPAN: not remember
SPAN: memory lapse
Tweet:
"""
is adderall a performance enhancing drug for mathletes?
"""
SPAN: null
---
Tweet:
"""
{tweet}
"""
'''


if __name__ == '__main__':
    dev_dataset_path = 'data/task1/Dev_2024/ade_extraction_gpt4'
    train_dataset_path = 'data/task1/Train_2024/ade_extraction_gpt4'
    test_dataset_path = 'data/task1/test/ade_extraction_gpt4'

    dataset_path = test_dataset_path
    span_from = int(sys.argv[1])
    span_to = int(sys.argv[2])

    os.makedirs(os.path.join(dataset_path, 'response'), exist_ok=True)
    with open(os.path.join(dataset_path, '..', 'tweets.json'), 'r') as f:
        dataset = json.load(f)

    with open(os.path.join(dataset_path, '..', 'classified.csv'), 'r') as f:
        reader = csv.DictReader(f)
        classified_tweets = []
        for line in reader:
            if not int(line['predicted']):
                continue
            classified_tweets.append(line['tweet_id'])

    dataset = {k:v for k, v in dataset.items() if k in classified_tweets}

    extracted_span_list = glob(os.path.join(dataset_path, 'response', '*.txt'))
    extracted_span_list = [os.path.splitext(os.path.basename(name))[0] for name in extracted_span_list]

    client = openai.OpenAI()

    for tweet_id, tweet in tqdm(list(dataset.items())[span_from:span_to]):
        if tweet_id in extracted_span_list:
            continue

        response = client.chat.completions.create(
          # model="gpt-3.5-turbo",
          model="gpt-4-turbo-2024-04-09",
          messages=[
            {
              "role": "user",
              "content": ade_mining.format(tweet=tweet)
            }
          ],
          temperature=0,
          max_tokens=64,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )

        with open(os.path.join(dataset_path, 'response', f'{tweet_id}.txt'), 'w') as f:
            f.write(response.choices[0].message.content)
