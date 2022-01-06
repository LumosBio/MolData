import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
# from docx import Document
from sklearn.decomposition import PCA
import seaborn as sns
from fuzzywuzzy import fuzz


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


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


df = pd.read_csv('merged_features_clustered_ner_cleaned.csv')
all_tagged_dfs = []
disease_clusters = []
target_clusters = []
total_unhighlighted = []
total_merged = []
for cluster_number in range(10):
    cluster_df = df[df['cluster'] == cluster_number]
    cluster_df.reset_index(inplace=True)
    descs = np.array(cluster_df['description'])
    # print(len(descs))

    document = Document('labeled/'+str(cluster_number)+'.docx')
    words = document._element.xpath('//w:r')

    all_words = []
    all_props = []
    for w in words:
        dummy = w.xml
        dummy = dummy.replace('<w:t xml:space=\"preserve\">', '<w:t>')
        if "<w:t>" not in dummy:
            continue
        current_word = dummy.split("<w:t>")[1].split("</w:t>")[0].lower().strip()
        if len(current_word) == 0:
            continue
        all_words.append(current_word)
        if "\"yellow\"" in dummy:
            all_props.append('disease')
        elif "FF0000" in dummy:
            all_props.append('target')
        else:
            all_props.append('normal')
    # print(len(all_props), len(all_words))

    text = document._element.xml
    text = text.replace('<w:t xml:space=\"preserve\">', '<w:t>')
    words = text.split('<w:t>')
    lines = []
    all_words2 = []
    all_paragraph = []
    paragraph_counter = 0
    high_light_counter = {}
    for w in range(len(words)):
        if "</w:t>" not in words[w]:
            continue
        current_word = words[w].split("</w:t>")[0].lower().strip()
        if len(current_word) != 0:
            all_words2.append(current_word)
            all_paragraph.append(paragraph_counter)
        if "\"yellow\"" in words[w]:
            high_light_counter[paragraph_counter] = 1
        if 'paraId' in words[w]:
            paragraph_counter += 1
        # if "by the brm gene" in words[w].lower():
        #     print(words[w], w)
    # print(len(all_words2))
    word_df = pd.DataFrame(data={'text': all_words,'text2': all_words2, 'paragraph': all_paragraph, 'properties': all_props})
    merged = []
    current_counter = 0
    dummy = ''
    dummy_disease = []
    dummy_target = []
    all_disease = []
    all_target = []
    previous_i = -1
    for i in range(len(all_words2)):
        if all_paragraph[i] != current_counter:
            current_counter = all_paragraph[i+1]
            merged.append(dummy)
            all_disease.append(dummy_disease)
            all_target.append(dummy_target)
            dummy = ''
            dummy_disease = []
            dummy_target = []
        if all_props[i] == 'disease':
            if '(' in all_words2[i] and not all_words2[i].startswith('('):
                current_words = all_words2[i].split('(')
                for c in current_words:
                    current_word = delete_punctuation(c)
                    dummy_disease.append(current_word.strip().lower())
            else:
                current_word = delete_punctuation(all_words2[i])
                dummy_disease.append(current_word.strip().lower())
        elif all_props[i] == 'target':
            # if '(' in all_words2[i] and not all_words2[i].startswith('('):
            #     current_words = all_words2[i].split('(')
            #     for c in current_words:
            #         current_word = delete_punctuation(c)
            #         dummy_target.append(current_word.strip().lower())
            # else:
            #     current_word = delete_punctuation(all_words2[i])
            #     dummy_target.append(current_word.strip().lower())
            # current_word = delete_punctuation(all_words2[i])
            # dummy_target.append(current_word.strip().lower())
            if previous_i == i - 1:
                # print(all_words2[i-1].strip().lower(), all_words2[i].strip().lower())
                dummy_target[-1] = dummy_target[-1] + all_words2[i].strip().lower()
                previous_i = i
            else:
                dummy_target.append(all_words2[i].strip().lower())
                previous_i = i
        dummy = dummy + all_words2[i] + ' '
    merged.append(dummy)
    all_disease.append(dummy_disease)
    all_target.append(dummy_target)
    print(cluster_number)
    print(len(all_target), len(all_disease), len(merged))
    print(len(cluster_df['description'].drop_duplicates()))
    print(len(high_light_counter))
    total_unhighlighted.append(len(merged) - len(high_light_counter))
    total_merged.append(len(merged))
    ordered_merged = []
    ordered_disease = []
    ordered_target = []
    match_scores = []
    mapping = {}
    for d in range(len(descs)):
        dummy = ''
        dummy_disease = []
        dummy_target = []
        overlap_counter = 0
        for m in range(len(merged)):
            current_counter = fuzz.token_set_ratio(merged[m], descs[d].lower())
            # current_counter = len(np.unique(intersection(np.unique(merged[m].split()), np.unique(descs[d].lower().split()))))
            if current_counter > overlap_counter:
                overlap_counter = current_counter
                dummy = merged[m]
                dummy_disease = all_disease[m]
                dummy_target = all_target[m]
                mapping[d] = m
        # match_scores.append(overlap_counter/len(descs[d].split()))
        match_scores.append(overlap_counter)
        # print(d, match_scores[-1])
        ordered_merged.append(dummy)
        ordered_disease.append(dummy_disease)
        ordered_target.append(dummy_target)

    ordered_disease_str = []
    ordered_target_str = []
    for i in range(len(ordered_disease)):
        dummy = ''
        for j in ordered_disease[i]:
            dummy += j + ', '
        ordered_disease_str.append(dummy.rstrip(', '))
        dummy = ''
        for j in ordered_target[i]:
            dummy += j + ', '
        ordered_target_str.append(dummy.rstrip(', '))

    # new_df = pd.DataFrame(data={'description': descs, 'new_description': ordered_merged, 'disease':ordered_disease_str, 'target': ordered_target_str})
    cluster_df['recovered_description'] = ordered_merged
    cluster_df['recovery_score'] = match_scores
    cluster_df['disease_tags_ground_truth'] = ordered_disease_str
    cluster_df['target_tags_ground_truth'] = ordered_target_str
    all_tagged_dfs.append(cluster_df)
    disease_clusters.append(ordered_disease)
    target_clusters.append(ordered_target)
merged_df = all_tagged_dfs[0]
for i in range(1, len(all_tagged_dfs)):
    merged_df = merged_df.merge(all_tagged_dfs[i], how='outer')

merged_df.to_csv('merged_features_clustered_ner_cleaned_extracted.csv', header=True, index=False)



disease_clusters_merged = []
disease_merged = []
for i in disease_clusters:
    dummy = []
    for j in i:
        for k in j:
            dummy.append(k)
    disease_clusters_merged.append(dummy)
    disease_merged.extend(dummy)

disease_counter = {}
for i in disease_merged:
    if i in disease_counter:
        disease_counter[i] += 1
    else:
        disease_counter[i] = 1

disease_counter = dict(sorted(disease_counter.items(), key=lambda item: item[1]))
x = list(disease_counter.keys())
y = list(disease_counter.values())
x.reverse()
y.reverse()
disease_counter_df = pd.DataFrame(data={'all_disease': x, 'count_all_disease': y})

all_x = []
all_y = []
for cluster_number in range(10):
    disease_counter = {}
    for i in disease_clusters_merged[cluster_number]:
        if i in disease_counter:
            disease_counter[i] += 1
        else:
            disease_counter[i] = 1

    disease_counter = dict(sorted(disease_counter.items(), key=lambda item: item[1]))
    x = list(disease_counter.keys())
    y = list(disease_counter.values())
    x.reverse()
    y.reverse()
    disease_counter_df_dummy = pd.DataFrame(data={str(cluster_number)+'_disease': x, str(cluster_number)+'_count': y})
    disease_counter_df = pd.concat([disease_counter_df,disease_counter_df_dummy], axis=1)

disease_counter_df.to_csv('count_all_diseases.csv', index=False, header=True)

target_clusters_merged = []
target_merged = []
for i in target_clusters:
    dummy = []
    for j in i:
        for k in j:
            dummy.append(k)
    target_clusters_merged.append(dummy)
    target_merged.extend(dummy)

target_counter = {}
for i in target_merged:
    if i in target_counter:
        target_counter[i] += 1
    else:
        target_counter[i] = 1

target_counter = dict(sorted(target_counter.items(), key=lambda item: item[1]))
x = list(target_counter.keys())
y = list(target_counter.values())
x.reverse()
y.reverse()
target_counter_df = pd.DataFrame(data={'all_target': x, 'count_all_target': y})

all_x = []
all_y = []
for cluster_number in range(10):
    target_counter = {}
    for i in target_clusters_merged[cluster_number]:
        if i in target_counter:
            target_counter[i] += 1
        else:
            target_counter[i] = 1

    target_counter = dict(sorted(target_counter.items(), key=lambda item: item[1]))
    x = list(target_counter.keys())
    y = list(target_counter.values())
    x.reverse()
    y.reverse()
    target_counter_df_dummy = pd.DataFrame(data={str(cluster_number)+'_target': x, str(cluster_number)+'_count': y})
    # target_counter_df[str(cluster_number)+'target'] = x
    # target_counter_df[str(cluster_number)+'count'] = y
    target_counter_df = pd.concat([target_counter_df,target_counter_df_dummy], axis=1)

target_counter_df.to_csv('count_all_targets.csv', index=False, header=True)

ner_results = np.array(merged_df['ner_tags'])
ner_results_cleaned = []
for n in ner_results:
    ner_results_cleaned.append(np.unique(n.strip('[]').replace('\'', '').split(',')))
ner_results_str = []
for n in ner_results_cleaned:
    dummy_tag = ''
    for t in n:
        dummy_tag += t.strip() + ','
    ner_results_str.append(dummy_tag.rstrip(','))
merged_df.pop('ner_tags')
merged_df['ner_tags'] = ner_results_str

merged_df.to_csv('merged_features_clustered_ner_cleaned_extracted.csv', header=True, index=False)

ner_results_str = np.array(pd.read_csv('merged_features_clustered_ner_cleaned_extracted.csv')['ner_tags'])
cluster_nums = np.array(pd.read_csv('merged_features_clustered_ner_cleaned_extracted.csv')['cluster'])

disease_merged = []
disease_clusters_merged = [[],[],[],[],[],[],[],[],[],[]]
for i in range(len(ner_results_str)):
    n = ner_results_str[i]
    if pd.isna(n):
        continue
    dummy = n.split(',')
    if len(dummy) > 0:
        disease_merged.extend(dummy)
        disease_clusters_merged[cluster_nums[i]].extend(dummy)

disease_counter = {}
for i in disease_merged:
    if i in disease_counter:
        disease_counter[i] += 1
    else:
        disease_counter[i] = 1

disease_counter = dict(sorted(disease_counter.items(), key=lambda item: item[1]))
x = list(disease_counter.keys())
y = list(disease_counter.values())
x.reverse()
y.reverse()
disease_counter_df = pd.DataFrame(data={'all_disease': x, 'count_all_disease': y})

all_x = []
all_y = []
for cluster_number in range(10):
    disease_counter = {}
    for i in disease_clusters_merged[cluster_number]:
        if i in disease_counter:
            disease_counter[i] += 1
        else:
            disease_counter[i] = 1

    disease_counter = dict(sorted(disease_counter.items(), key=lambda item: item[1]))
    x = list(disease_counter.keys())
    y = list(disease_counter.values())
    x.reverse()
    y.reverse()
    disease_counter_df_dummy = pd.DataFrame(data={str(cluster_number)+'_disease': x, str(cluster_number)+'_count': y})
    # disease_counter_df[str(cluster_number)+'disease'] = x
    # disease_counter_df[str(cluster_number)+'count'] = y
    disease_counter_df = pd.concat([disease_counter_df,disease_counter_df_dummy], axis=1)

disease_counter_df.to_csv('count_all_diseases_ner.csv', index=False, header=True)

import pandas as pd
import numpy as np
annot_df = pd.read_csv('key.csv')
annot_classes = list(np.unique(np.array(annot_df['class'])))
annot_map = {}
for a in range(len(annot_classes)):
    annot_map[a] = annot_classes[a]

disease_class = []
for a in annot_classes:
    current_df = annot_df[annot_df['class'] == a]
    current_diseases = list(np.unique(np.array(current_df['name'])))
    disease_class.append(current_diseases)
annot_diseases = list(np.unique(np.array(annot_df['name'])))
df = pd.read_csv('merged_features_clustered_ner_cleaned_extracted.csv')
diseases = np.array(df['disease_tags_ground_truth'])
unannotated = []
benchmark_tag = []
for i in range(len(diseases)):
    n = diseases[i]
    if pd.isna(n):
        benchmark_tag.append('')
        continue
    dummy = n.split(',')
    tag_dummy = []
    if len(dummy) == 0:
        benchmark_tag.append('')
        continue
    for dc in dummy:
        disease = dc.strip()
        for d_class in range(len(disease_class)):
            if disease in disease_class[d_class]:
                if annot_map[d_class] not in tag_dummy:
                    tag_dummy.append(annot_map[d_class])
            if disease not in annot_diseases:
                if disease not in unannotated:
                    unannotated.append(disease)
                    print(disease)
    tag_dummy_str = ''
    for tag_d in tag_dummy:
        tag_dummy_str += tag_d + ','
    benchmark_tag.append(tag_dummy_str.rstrip(','))
df['benchmark_tag'] = benchmark_tag
df.to_csv('merged_features_clustered_ner_cleaned_extracted_benchmarked.csv', header=True, index=False)

dummy = df[df['benchmark_tag'] == '']

print(len(dummy), len(dummy.dropna(subset=['disease_tags_ground_truth'])),len(dummy.dropna(subset=['target_tags_ground_truth'])), len(df))
print(np.sum(total_unhighlighted))
print(np.sum(total_merged))
for i in range(10):
    print('Cluster', str(i), 'has', str(total_unhighlighted[i]), 'not highlighted out of the total',
          str(total_merged[i]), 'paragraphs and', str(len(dummy[dummy['cluster'] == i])),
          'untagged descriptions out of', str(len(df[df['cluster'] == i])), 'descriptions.')
print('Total', str(np.sum(total_unhighlighted)), 'not highlighted paragraphs and', str(len(dummy)),
      'untagged descriptions and', str(len(df)-len(dummy)), 'tagged descriptions.')




useless_tags = [
'gpr119',
'damage',
'all',
'aml',
'syndrome',
'mbt',
'anthelmintics',
'antithrombic',
'embryonic lethality',
'transcriptional and translational regulation',
'sds',
'type iv',
'ftld',
'epigenetics',
'hnf4',
'metabolic',
'neonatal',
'hh',
'autophagy',
'colorectal',
'trust',
'ra',
'cellular proliferation',
'liver',
'mpds',
'adult bone formation',
'firefly',
'firefly luciferase',
'luminescent',
'diseases',
'disease',
'dyrk1 kinase',
'cell survival',
'mv',
'of mycobacterium tuberculosis',
'diarrhea',
'mas',
'at',
'gsdii',
'omim 230400',
'muscle',
'pca',
'proliferation stimuli',
'ad',
'cf',
'luciferase',
'gpcr',
'cjd',
'hd',
'skeletal muscle',
'frda',
'hcs',
'mm',
'cll',
'cellular senescence',
'tyrosine kinases',
'kappab',
'cml',
'drg',
'alqts',
'relapse',
'vascular smooth muscle',
'muscle diseases',
'cytotoxic',
'tgfbeta antagonists',
'pxrluc',
'liver regeneration'
]
for u in unannotated:
    if u not in useless_tags:
        print(u)

# unused_aids.extend(list(df[df['benchmark_tag'] == '']['AID']))
unused_aids = [
 588856,
 588855,
 1663,
 2216,
 1832,
 782,
 588342,
 1865,
 2599,
 540295,
 540308,
 720647,
 743238,
 588674,
 602363,
 651704,
 651658,
 488862,
 504414,
 652115,
 504441,
 504408,
 602252,
 485317,
 2288,
 2289,
 2629,
 1875,
 2094,
 2098,
 2563,
 588478,
 1159583,
 485294,
 485341,
 1721,
 1722,
 651999,
 2805,
 2806,
 434973,
 2524,
 2540,
 2544,
 1016,
 1006,
 1020,
 1027,
 1136,
 720516]
used_aids = [aid for aid in df['AID'] if aid not in unused_aids]
print(len(df))
df_cleaned = df[df['AID'].isin(used_aids)]
print(len(df_cleaned))
df_cleaned.to_csv('merged_features_clustered_ner_cleaned_extracted_benchmarked_cleaned.csv', header=True, index=False)

