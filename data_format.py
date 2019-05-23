
import pandas as pd

df = pd.read_csv('../news_summary.csv', encoding = 'latin1')

print(df.columns)

print(df.shape)

# print("--------------")
# print(df['ctext'][0])
# print("--------------")
# print(df['text'][0])
# print("--------------")

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

f1 = open("input.txt", encoding='utf-8', mode = 'w')
f2 = open("output.txt", encoding='utf-8', mode =  "w")
for doc, sum in zip(df['ctext'],df['text']):
    sents = sent_tokenize(str(doc))
    sents_sum = sent_tokenize(str(sum))
    lines1 = []
    lines2 = []
    for i in range(len(sents)):
        sent1 = word_tokenize(sents[i])
        lines1.append(" ".join(sent1))
    for i in range(len(sents_sum)):
        sent2 = word_tokenize(sents_sum[i])
        lines2.append(" ".join(sent2))
    f1.write(" ".join(lines1) + "\n")
    f2.write(" ".join(lines2) + "\n")

f1.close()
f2.close()
