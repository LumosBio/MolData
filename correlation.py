import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

mol_df = pd.read_csv('merged_cleaned_benchmarked_threshold_scaffold_split_stratified_pca.csv')
activity_columns = [c for c in list(mol_df.columns) if 'activity' in c]

corr = mol_df[activity_columns].corr()
corr.to_csv('results/correlation_all.csv', header=True, index=True)

corr = pd.read_csv('results/correlation_all.csv', index_col=0)
map_df = pd.read_csv('aid_tag_mapping.csv')
desc_df = pd.read_csv('merged_features_clustered_ner_cleaned_extracted_benchmarked_cleaned_counted_threshold.csv')
desc_info = {}
for i in range(len(desc_df)):
    desc_info['activity_' + str(desc_df.iloc[i]['AID'])] = [desc_df.iloc[i]['name'], desc_df.iloc[i]['source'], desc_df.iloc[i]['target'], desc_df.iloc[i]['benchmark_tag']]
all_categories = ['all'] + list(map_df.columns)[1:]
for current_tag in all_categories:
    # current_tag = 'toxicity'
    print(current_tag)
    if current_tag == 'all':
        current_corr = corr
    else:
        current_aids = np.array(list(map_df[map_df[current_tag] == 1]['AID']))
        current_corr = corr[current_aids]
        current_corr = current_corr[current_corr.index.isin(current_aids)]
        corr.to_csv('results/correlation_'+current_tag+'.csv', header=True, index=False)
    if current_tag in ['all','cancer']:
        fontsize = 1
    else:
        fontsize = 4
    current_columns = [c.lstrip('activity_') for c in list(current_corr.columns)]
    current_corr = current_corr.reindex(list(current_corr.columns))
    # current_corr.to_csv('toxicity_correlation_matrix.csv', index=True)
    # current_corr = np.array(current_corr)
    # dummy = []
    # for i in range(len(current_corr)):
    #     for j in range(len(current_corr)):
    #         dummy.append([i,j,current_corr[i,j], current_columns[i], current_columns[j]])
    # dummy = np.array(dummy)
    # dummy_df = pd.DataFrame(dummy, columns=['x', 'y', 'correlation', 'AID_x', 'AID_y'])
    # dummy_df.to_csv('toxicity_correlation_matrix.csv', index=False)
    fig = plt.figure(figsize=(8,4), dpi=300)
    ax = fig.add_subplot(111)
    cax = ax.matshow(current_corr,cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(current_columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    # plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax.set_xticklabels(current_columns, fontsize=fontsize)
    ax.set_yticklabels(current_columns, fontsize=fontsize)
    plt.xlabel('Bioassay AID', fontsize=fontsize+7)
    ax.xaxis.set_label_position('top')
    plt.ylabel('Bioassay AID', fontsize=fontsize+7)
    plt.tight_layout()
    plt.savefig('results/correlation_'+current_tag+'.png', format='png', dpi=300)
    plt.show()
    plt.close()

corr_array = np.array(corr)
corr_array_abs = np.absolute(np.array(corr))
all_aids = list(corr.columns)
threshold = 0.5
interesting_aids = []
interesting_corr = []
for i in range(len(corr_array)):
    first_aid = all_aids[i]
    first_source = desc_info[first_aid][1]
    for j in range(len(corr_array)):
        if i == j:
            continue
        second_aid = all_aids[j]
        second_source = desc_info[second_aid][1]
        if first_source == 'Tox21' or second_source == 'Tox21':
            continue
        current_corr = corr_array[i, j]
        if abs(current_corr) >= threshold:
            if abs(int(first_aid.lstrip('activity_')) - int(second_aid.lstrip('activity_'))) > 5:
                if first_source == second_source:
                    source_overlap = 1
                else:
                    source_overlap = 0
                if first_aid not in interesting_aids:
                    interesting_aids.append(first_aid)
                if second_aid not in interesting_aids:
                    interesting_aids.append(second_aid)
                interesting_corr.append([first_aid, second_aid, current_corr, source_overlap])
print(len(interesting_aids))
interesting_corr = pd.DataFrame(data=interesting_corr, columns=['first_AID', 'second_AID', 'corr', 'same_source'])
interesting_corr = interesting_corr.sort_values(by=['corr'], ascending=False)
added_info = []
for i in range(len(interesting_corr)):
    added_info.append(desc_info[interesting_corr.iloc[i]['first_AID']] + desc_info[interesting_corr.iloc[i]['second_AID']])
interesting_corr[['first_name', 'first_source', 'first_target', 'first_benchmark_tag', 'second_name', 'second_source', 'second_target', 'second_benchmark_tag']] = np.array(added_info)
interesting_corr_unique = interesting_corr.iloc[list(np.arange(0,len(interesting_corr), 2))]


overlap = []
active_overlap = []
sizes = []
actives = []
for i in range(len(interesting_corr_unique)):
    first_aid = interesting_corr_unique.iloc[i]['first_AID']
    second_aid = interesting_corr_unique.iloc[i]['second_AID']
    current_df = mol_df[[first_aid, second_aid]].dropna(how='any')
    overlap.append(len(current_df))
    current_df_1 = current_df[current_df[first_aid] == 1]
    current_df_2 = current_df[current_df[second_aid] == 1]
    actives.append([len(current_df_1), len(current_df_2)])
    current_df = current_df[current_df[first_aid] == 1]
    current_df = current_df[current_df[second_aid] == 1]
    active_overlap.append(len(current_df))
    sizes.append([len(mol_df[first_aid].dropna(how='any')), len(mol_df[second_aid].dropna(how='any'))])


interesting_corr_unique.insert(loc=3, column='overlap', value=overlap)
interesting_corr_unique[['first_size', 'second_size']] = sizes
interesting_corr_unique[['first_active', 'second_active']] = actives

interesting_corr_unique.insert(loc=4, column='overlap_active', value=active_overlap)
expert_df = pd.read_csv('results/correlation_interesting_expert.csv')
previous = []
for i in range(len(expert_df)):
    first_aid = expert_df.iloc[i]['first_AID']
    second_aid = expert_df.iloc[i]['second_AID']
    previous.append([first_aid, second_aid])
    previous.append([second_aid, first_aid])
previous_checked = []
for i in range(len(interesting_corr_unique)):
    first_aid = interesting_corr_unique.iloc[i]['first_AID']
    second_aid = interesting_corr_unique.iloc[i]['second_AID']
    if [first_aid, second_aid] in previous:
        previous_checked.append(1)
    else:
        previous_checked.append(0)
interesting_corr_unique.insert(loc=3, column='previously_checked', value=previous_checked)

interesting_corr_unique.to_csv('results/correlation_interesting.csv', header=True, index=False)
# interesting_corr_unique.to_csv('results/correlation_interesting_diff_source.csv', header=True, index=False)

expert_df = pd.read_csv('results/correlation_interesting_expert.csv')

overlap = []
active_overlap = []
sizes = []
actives = []
for i in range(len(expert_df)):
    first_aid = expert_df.iloc[i]['first_AID']
    second_aid = expert_df.iloc[i]['second_AID']
    current_df = mol_df[[first_aid, second_aid]].dropna(how='any')
    overlap.append(len(current_df))
    current_df_1 = current_df[current_df[first_aid] == 1]
    current_df_2 = current_df[current_df[second_aid] == 1]
    actives.append([len(current_df_1), len(current_df_2)])
    current_df = current_df[current_df[first_aid] == 1]
    current_df = current_df[current_df[second_aid] == 1]
    active_overlap.append(len(current_df))
    sizes.append([len(mol_df[first_aid].dropna(how='any')), len(mol_df[second_aid].dropna(how='any'))])
# a = mol_df['activity_1259404']
# len(a.dropna(how='any'))

expert_df.insert(loc=3, column='overlap', value=overlap)
expert_df[['first_size', 'second_size']] = sizes
expert_df[['first_active', 'second_active']] = actives

expert_df.insert(loc=4, column='overlap_active', value=active_overlap)

# interesting_corr_unique.to_csv('results/correlation_interesting.csv', header=True, index=False)
expert_df.to_csv('results/correlation_interesting_expert_overlap.csv', header=True, index=False)

current_corr = corr[interesting_aids]
current_corr = current_corr[current_corr.index.isin(interesting_aids)]
current_columns = [c.lstrip('activity_') for c in list(current_corr.columns)]
current_corr = current_corr.reindex(list(current_corr.columns))
fig = plt.figure(figsize=(8,4), dpi=300)
ax = fig.add_subplot(111)
cax = ax.matshow(current_corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(current_columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(current_columns, fontsize=5)
ax.set_yticklabels(current_columns, fontsize=5)
plt.tight_layout()
# plt.savefig('results/correlation_interesting.png', format='png', dpi=300)
plt.savefig('results/correlation_interesting_diff_source.png', format='png', dpi=300)
plt.show()
plt.close()


final_df = pd.read_csv('results/correlation_interesting_final.csv')
first_aids = list(final_df['first'])
second_aids = list(final_df['second'])
final_aids = []
for i in range(len(first_aids)):
    if first_aids[i] not in final_aids:
        final_aids.append(first_aids[i])
    if second_aids[i] not in final_aids:
        final_aids.append(second_aids[i])


current_corr = corr[final_aids]
current_corr = current_corr[current_corr.index.isin(final_aids)]
current_columns = [c.lstrip('activity_') for c in list(current_corr.columns)]
current_corr = current_corr.reindex(list(current_corr.columns))
#
# current_corr = np.array(current_corr)
# dummy = []
# for i in range(len(current_corr)):
#     for j in range(len(current_corr)):
#         dummy.append([i,j,current_corr[i,j], current_columns[i], current_columns[j]])
# dummy = np.array(dummy)
# dummy_df = pd.DataFrame(dummy, columns=['x', 'y', 'correlation', 'AID_x', 'AID_y'])
# dummy_df.to_csv('interesting_correlation_matrix.csv', index=False)

fig = plt.figure(figsize=(8,4), dpi=300)
ax = fig.add_subplot(111)
cax = ax.matshow(current_corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(current_columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(current_columns, fontsize=5)
ax.set_yticklabels(current_columns, fontsize=5)
plt.xlabel('Bioassay AID', fontsize=5 + 7)
ax.xaxis.set_label_position('top')
plt.ylabel('Bioassay AID', fontsize=5 + 7)
plt.tight_layout()
# plt.savefig('results/correlation_interesting.png', format='png', dpi=300)
plt.savefig('results/correlation_interesting_final.png', format='png', dpi=300)
plt.show()
plt.close()

activity_columns = [c for c in list(mol_df.columns) if 'activity' in c]
all_sums = mol_df[activity_columns].sum(axis=1)
all_counts = mol_df[activity_columns].count(axis=1)
plt.hist(all_sums, bins=range(10,141))
plt.hist(all_counts, bins='auto')
count_df = pd.DataFrame(data=all_sums, columns=['active'])
count_df['total'] = all_counts
count_df['activity_percentage'] = 100 * count_df['active']/count_df['total']

plt.scatter(count_df['total'], count_df['active'])

from scipy.stats import gaussian_kde
x1 = np.array(count_df['total'])
y1 = np.array(count_df['active'])
xy1 = np.vstack([x1, y1])
z1 = gaussian_kde(xy1)(xy1)

len(z1)

efficacy_df = pd.DataFrame(data=x1, columns=["Number of Screens"])
efficacy_df["Number of Active Results"] = y1
efficacy_df["Density"] = z1
efficacy_df.to_csv("gaussian_efficacy.csv", index=False)
smaller_efficacy_df = efficacy_df.sample(n=200000, random_state=42)
smaller_efficacy_df.to_csv("gaussian_efficacy_small.csv", index=False)

x1 = np.array(count_df['total'])
y1 = np.array(count_df['activity_percentage'])
xy1 = np.vstack([x1, y1])
z1 = gaussian_kde(xy1)(xy1)

efficacy_label = []
for i in range(len(count_df)):
    current_data = count_df.iloc[i]
    current_total = current_data['total']
    current_active = current_data['active']
    current_percent = current_data['activity_percentage']
    if current_percent <= 2:
        efficacy_label.append('Wasted')
    elif current_active >= 50:
        efficacy_label.append('Possible Toxic')
    elif current_total > 100:
        efficacy_label.append('Familiar Molecules')
    else:
        efficacy_label.append('New Molecules')

toxic_line = []
for i in range(50,600,10):
    toxic_line.append([int(i), int(5000/i)])
import pandas as pd
toxic_line_df = pd.DataFrame(data=toxic_line, columns=['x','y'])
toxic_line_df.to_csv("toxic_line.csv", header=True, index=False)

efficacy_df = pd.DataFrame(data=x1, columns=["Number of Screens"])
efficacy_df["Active Percentage"] = y1
efficacy_df["Density"] = z1
efficacy_df["Efficacy Type"] = efficacy_label
efficacy_df.to_csv("gaussian_efficacy_percentage.csv", index=False)
smaller_efficacy_df = efficacy_df.sample(n=300000, random_state=42)
smaller_efficacy_df.to_csv("gaussian_efficacy_percentage_small.csv", index=False)

all_sources = np.array(desc_df['source'])
all_aids = np.array(desc_df['AID'])
# aid_source = {'activity_' + str(s):[] for s in all_aids}
aid_source = {}
for i in range(len(all_aids)):
    aid_source['activity_' + str(all_aids[i])] = all_sources[i]

# for s in aid_source:
#     print(len(aid_source[s]))

source_index = {}
unique_sources = np.unique(all_sources)
for s in range(len(unique_sources)):
    source_index[unique_sources[s]] = s

mol_source = []
for m in range(len(mol_df)):
    current_data = mol_df.iloc[m][activity_columns]
    current_aids = list(current_data[pd.notna(current_data)].index)
    current_sources = np.unique([source_index[aid_source[aid]] for aid in current_aids])
    mol_source.append(current_sources)
    if m % 100000 == 0:
        print(m)

source_connection = np.zeros((len(unique_sources), len(unique_sources)), dtype=int)
counter = 0
for s in mol_source:
    if len(s) == 1:
        source_connection[int(s[0]), int(s[0])] += 1
    else:
        for i in range(len(s)):
            if i == len(s) - 1:
                break
            for j in range(i + 1, len(s)):
                source_connection[int(s[i]), int(s[j])] += 1
                source_connection[int(s[j]), int(s[i])] += 1
    if counter % 10000 == 0:
        print(counter)
    counter += 1

source_con_df = pd.DataFrame(data=source_connection, columns=unique_sources)
source_con_df.insert(loc=0, column='source', value=unique_sources)
source_con_df.to_csv('source_molecular_overlap.csv', header=True, index=False)