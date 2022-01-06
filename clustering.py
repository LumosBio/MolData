from biobert_embedding.embedding import BiobertEmbedding
import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
# from docx import Document
from sklearn.decomposition import PCA
import seaborn as sns

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def sentence_vector(tokenized_text, biobert):
    encoded_layers = biobert.eval_fwdprop_biobert(tokenized_text)

    # `encoded_layers` has shape [12 x 1 x 22 x 768]
    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = encoded_layers[11][0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding


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


def delete_punctuation(sample):
    sample = sample.replace('-', '')
    sample = sample.replace('/', '')
    sample = sample.replace('(', '')
    sample = sample.replace(')', '')
    sample = sample.replace('\'', '')
    sample = sample.replace('.', '')
    sample = sample.replace(':', '')
    sample = sample.replace(',', '')
    sample = sample.replace(';', '')
    sample = sample.replace('_', '')
    return sample


df = pd.read_csv('merged.csv')
df = df[df['substance_num'] >= 100]
df_tox = df[df['source'] == 'Tox21']
df = df[df['substance_num'] >= 100000]
df = df.merge(df_tox, how='outer')
biobert = BiobertEmbedding()

sources = np.array(df['source'])
print(np.unique(sources))


embs = []
long_counter = 0
descs = np.array(df['description'])
titles = np.array(df['name'])
# Extract embeddings for descriptions
for desc in descs:
    desc = biobert.process_text(desc.lower())
    embs.append(sentence_vector(desc[:512], biobert))
    if len(desc) > 512:
        long_counter += 1
print(long_counter, 'out of', len(descs), 'descriptions were truncated (Max 512 tokens).')

embs_np = []
for e in embs:
    embs_np.append(e.numpy())
embs_np = np.array(embs_np)
print(embs_np.shape)

# Extract embeddings for titles
embs_title = []
long_counter = 0
for title in titles:
    title = biobert.process_text(title.lower())
    embs_title.append(sentence_vector(title[:512], biobert))
    if len(title) > 512:
        long_counter += 1
print(long_counter, 'out of', len(titles), 'descriptions were truncated (Max 512 tokens).')

embs_title_np = []
for e in embs_title:
    embs_title_np.append(e.numpy())
embs_title_np = np.array(embs_title_np)
print(embs_title_np.shape)

# Concatenate embeddings for both titles and descriptions
features = np.concatenate((embs_np, embs_title_np), axis=1)
print(features.shape)

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans_kwargs = {
"init": "k-means++",
 "n_init": 10,
"max_iter": 1000,
"random_state": 42}

# Cluster the text features and find optimum number of clusters
sse = []
total_k = 51
for k in range(1, total_k):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)
    print(k)
plt.style.use("fivethirtyeight")
plt.plot(range(1, total_k), sse)
plt.xticks(range(1, total_k))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.tight_layout()
plt.savefig('knee.png', format='png', dpi=300)
plt.show()

kl = KneeLocator(range(1, total_k), sse, curve="convex", direction="decreasing")
optimum_k = kl.elbow
print(optimum_k)
kmeans = KMeans(n_clusters=optimum_k, **kmeans_kwargs)
kmeans.fit(scaled_features)

# Perform PCA to be able to display the clusters
pca = PCA(n_components=2, random_state=42)
pca_features = pca.fit_transform(scaled_features)
label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(sources)

df['cluster'] = kmeans.labels_
for i in range(len(features[0])):
    df['feature'+str(i)] = features[:,i]

# Add cluster information to the data
df.to_csv('merged_features_clustered.csv', header=True, index=False)

# Display the clusters
pcadf = pd.DataFrame(pca_features,columns=["Principal Component 1", "Principal Component 2"])
pcadf["Cluster"] = kmeans.labels_
pcadf["Data Source"] = label_encoder.inverse_transform(true_labels)

pcadf = pcadf.replace({'Data Source': {'Broad Institute': 'Broad Institute', 'Burnham Center for Chemical Genomics': 'Burnham Center',
 'Emory University Molecular Libraries Screening Center': 'Emory University',
 'ICCB-Longwood Screening Facility, Harvard Medical School': 'ICCB-Longwood',
 'Johns Hopkins Ion Channel Center': 'Johns Hopkins', 'NMMLSC':'NMMLSC',
 'National Center for Advancing Translational Sciences (NCATS)': 'NCATS',
 'The Scripps Research Institute Molecular Screening Center': 'Scripps', 'Tox21': 'Tox21'}})

# plt.style.use("fivethirtyeight")
plt.style.use("default")

plt.figure(figsize=(10, 8))
# fix color wheel
scat = sns.scatterplot("Principal Component 1", "Principal Component 2", s=100,data=pcadf,
                       hue="Cluster",style="Data Source", palette=sns.color_palette("tab10",len(np.unique(df['cluster']))))
# scat.set_title("Clustering results")
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.xlabel("Principal Component 1", fontsize=20)
plt.ylabel("Principal Component 2", fontsize=20)
plt.tick_params(axis="y",direction="in")
plt.tick_params(axis="x",direction="in")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=10, ncol=2, bbox_to_anchor=(0.44, 0.62))
# plt.legend(ncol=2)
plt.tight_layout()
plt.savefig('clusters.png', format='png')
plt.show()

###############################################################




