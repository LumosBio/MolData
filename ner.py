from transformers.pipelines import pipeline
import transformers
import torch
import pandas as pd
import numpy as np
print(transformers.__version__)
print(torch.__version__)

classifier = pipeline("ner", model='alvaroalon2/biobert_diseases_ner')
df = pd.read_csv('merged.csv')
df = df[df['substance_num'] >= 100]
df_tox = df[df['source'] == 'Tox21']
df = df[df['substance_num'] >= 100000]
df = df.merge(df_tox, how='outer')
# biobert = BiobertEmbedding()

sources = np.array(df['source'])
print(np.unique(sources))
def clean_punctuation(sample):
    sample = sample.replace('-', ' ')
    sample = sample.replace('/', ' ')
    sample = sample.replace('(', ' ')
    sample = sample.replace(')', ' ')
    sample = sample.replace('\'', ' ')
    sample = sample.replace('.', ' ')
    sample = sample.replace(':', ' ')
    sample = sample.replace(',', ' ')
    sample = sample.replace(';', ' ')
    sample = sample.replace('_', ' ')
    return sample
descs = np.array(df['description'])
titles = np.array(df['name'])
descs = np.array([clean_punctuation(d) for d in descs])
titles = np.array([clean_punctuation(d) for d in titles])


disease_tags = []
for i in range(len(descs)):
    dummy = titles[i].lower() + ' . ' + descs[i].lower()
    # dummy = dummy.split()
    ner = classifier(dummy)
    disease_dummy = []
    for j in ner:
        if 'DISEASE' in j['entity']:
            w = j['word']
            if '##' in w and len(disease_dummy) != 0:
                disease_dummy[-1] = disease_dummy[-1] + w.lstrip('##')
            else:
                disease_dummy.append(w)
    disease_tags.append(disease_dummy)
df = pd.read_csv('merged_features_clustered.csv')
print(len(df), len(disease_tags))


df.insert(10, 'ner_tags', disease_tags)
df.to_csv('merged_features_clustered_ner.csv')

df = pd.read_csv('merged_features_clustered_ner.csv')
for column in df.columns:
    if 'feature' in column or 'Unnamed' in column:
        df.drop(column, axis=1, inplace=True)
df.to_csv('merged_features_clustered_ner_cleaned.csv', header=True, index=False)

