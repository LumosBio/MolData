import pandas as pd
import numpy as np
from rdkit.Chem import MolFromSmiles, MolToSmiles
np.random.seed(123)


def canon_smile(smile):
    return MolToSmiles(MolFromSmiles(smile), isomericSmiles=True)


aids = np.array(pd.read_csv('merged_features_clustered_ner_cleaned_extracted_benchmarked.csv')['AID'])

str_aid = ''
for aid in aids:
    str_aid+='https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi?query=download&record_type=datatable&actvty=all&response_type=save&aid='+str(aid) + '\n'

text_file = open('download_links.txt', "w+")
n = text_file.write(str_aid)
text_file.close()

all_cids = []
counter = 0
for aid in aids:
    all_cids.extend(np.array(pd.read_csv('datasets/AID_'+str(aid)+'_datatable_all.csv', usecols=['PUBCHEM_CID'])['PUBCHEM_CID'], dtype=int))
    print(counter, aid)
    counter += 1

print(len(all_cids))
unique_cids = np.unique(np.array(all_cids))
print(len(unique_cids))

for chunk in range(3):
    str_cid = ''
    for cid in unique_cids[499999 * chunk: 499999 * (chunk + 1)]:
        str_cid += str(cid) + '\n'
    text_file = open('unique_cids_'+str(chunk)+'.txt', "w+")
    n = text_file.write(str_cid)
    text_file.close()


li = []
for filename in range(3):
    df = pd.read_csv('smiles/'+str(filename)+'.txt', delimiter='\t')
    li.append(df)
smiles_df = pd.concat(li, axis=0, ignore_index=True)


canon_list = []
counter = 0
error_counter = 0
all_smiles = np.array(smiles_df['smiles'])
for s in all_smiles:
    try:
        cannon_dummy = canon_smile(s)
    except:
        cannon_dummy = ''
        error_counter += 1
    canon_list.append(cannon_dummy)
    if counter % 100000 == 0:
        print(counter, error_counter)
    counter += 1
smiles_df['canon_smiles'] = np.array(canon_list)
smiles_df.to_csv('smiles/canon_map.csv', header=True, index=False)







aids = np.array(pd.read_csv('merged_features_clustered_ner_cleaned_extracted_benchmarked.csv')['AID'])
all_df = []
for aid in aids:
    df = pd.read_csv('datasets/AID_'+str(aid)+'_datatable_all.csv', low_memory=False, usecols=['PUBCHEM_RESULT_TAG'])
    correct_column = 3
    for i in range(8):
        if df['PUBCHEM_RESULT_TAG'][i] == '1':
            correct_column = i
    print(aid, correct_column)
    all_df.append(pd.read_csv('datasets/AID_'+str(aid)+'_datatable_all.csv', low_memory=False, skiprows=list(range(1,correct_column+1)),
                     usecols=lambda c: c in set(['PUBCHEM_ACTIVITY_OUTCOME', 'PUBCHEM_CID', 'Phenotype', 'Activity Summary'])))

canon_df = pd.read_csv('smiles/canon_map.csv')
canon_map = pd.Series(canon_df['canon_smiles'].values,index=canon_df['PUBCHEM_CID']).to_dict()


all_out_keys = ['Active', 'Inactive', 'Inconclusive', 'Unspecified']
all_pheno_keys = ['Activator', 'Active', 'Cytotoxic', 'Fluorescent', 'Inactive', 'Inconclusive', 'Inhibitor',
                  'Quencher', 'Signal activator', 'Signal inhibitor', 'ikB active']
all_summary_keys = ['active agonist', 'active antagonist', 'inactive', 'inconclusive', 'inconclusive agonist',
                    'inconclusive agonist (cytotoxic)', 'inconclusive agonist (fluorescent)', 'inconclusive antagonist',
                    'inconclusive antagonist (cytotoxic)', 'inconclusive antagonist (fluorescent)']

def check_type_exist(type_, dict_):
    if type_ in dict_:
        return dict_[type_]
    else:
        return 0


all_count_type = []
all_df_cleaned = []
# flag = 0
for file_counter in range(len(all_df)):
    # file_counter = 0
    df = all_df[file_counter]
    print(aids[file_counter])
    print('Initial shape: ', df.shape)
    # print(df.columns)

    # Delete empty or duplicate smiles
    df = df.dropna(subset=['PUBCHEM_CID'])
    df = df.drop_duplicates(subset='PUBCHEM_CID', keep='first')
    # df.reset_index(inplace=True)
    print('Shape after deleting empty or duplicate smiles: ', df.shape)

    df = df[df['PUBCHEM_ACTIVITY_OUTCOME'] != 'Inconclusive']
    df = df[df['PUBCHEM_ACTIVITY_OUTCOME'] != 'Unspecified']
    print('Shape after deleting iconclusive and unspecified: ', df.shape)
    #     df = df[df[phenotype_column] != 'Cytotoxic']

    # Find unique values in columns
    df_dropped = df.dropna(subset=['PUBCHEM_ACTIVITY_OUTCOME'])
    phenotype_column = None
    outcome_column = None
    summary_column = None
    pheno_dict = {}
    outcome_dict = {}
    summary_dict = {}
    for column in df.columns:
        if 'Phenotype' in column:
            phenotype_column = column
            # print(column, np.unique(df_dropped[column]))
            pheno_dict = df[phenotype_column].value_counts().to_dict()
            # all_pheno_keys.extend(pheno_dict.keys())
            print(pheno_dict)
        if 'OUTCOME' in column:
            outcome_column = column
            # print(column, np.unique(df_dropped[column]))
            outcome_dict = df[outcome_column].value_counts().to_dict()
            # all_out_keys.extend(outcome_dict.keys())
            print(outcome_dict)
        if 'Summary' in column:
            summary_column = column
            # print(column, np.unique(df_dropped[column]))
            summary_dict = df[summary_column].value_counts().to_dict()
            # all_summary_keys.extend(summary_dict.keys())
            print(summary_dict)
    count_type = []
    for k in all_out_keys:
        count_type.append(check_type_exist(k, outcome_dict))
    for k in all_pheno_keys:
        count_type.append(check_type_exist(k, pheno_dict))
    for k in all_summary_keys:
        count_type.append(check_type_exist(k, summary_dict))
    all_count_type.append(count_type)

    df[outcome_column] = df[outcome_column].replace({'Active': 1, 'Inactive': 0})
    # print(df.columns)
    df.rename({outcome_column: 'activity_'+str(aids[file_counter])}, axis=1, inplace=True)
    df['smiles'] = df['PUBCHEM_CID'].map(canon_map)

    # Delete duplicates
    df = df.drop_duplicates(subset='smiles', keep='first')
    # df.reset_index(inplace=True)
    print('Shape after deleting duplicate canon smiles: ', df.shape)

    # Shuffle and save
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    all_df_cleaned.append(df)
    df.to_csv('cleaned_datasets/'+ str(aids[file_counter])+'_cleaned.csv', header=True, index=False)
all_count_type = np.array(all_count_type)
count_df = pd.DataFrame(data=all_count_type, columns=all_out_keys+all_pheno_keys+all_summary_keys)
count_df.insert(0, 'AID', aids)
count_df.to_csv('type_count_wo_inconclusive.csv', index=False, header=True)


all_df = [pd.read_csv('cleaned_datasets/'+ str(aid)+'_cleaned.csv', low_memory=False) for aid in aids]
for file_counter in range(len(all_df)):
    df = all_df[file_counter]
    outcome_column = None
    for column in df.columns:
        if 'activity' in column:
            outcome_column = column
            # print(column, np.unique(df[column]))
    # if len(np.unique(df[outcome_column])) == 0:
    #     continue
    df = df[['smiles', outcome_column]]
    if file_counter == 0:
        merged = df
    else:
        merged = merged.merge(df, on=['smiles'], how='outer')
    print(file_counter)
print(merged.head())
print(merged.shape)
print(merged.iloc[0])
smiles_canon = merged['smiles']
# merged.nunique(axis=0)
# print('smiles and unique smiles: ', len(smiles_canon), len(np.unique(smiles_canon)))
print('number of rows with missing data: ', merged.shape[0] - merged.dropna().shape[0])
canon_map_reversed = {}
for k in canon_map:
    if canon_map[k] not in canon_map_reversed:
        canon_map_reversed[canon_map[k]] = str(k)
    else:
        canon_map_reversed[canon_map[k]] += ','+str(k)
cleaned_ids = []
for s in smiles_canon:
    cleaned_ids.append(canon_map_reversed[s])
merged.insert(1,'PUBCHEM_CID', np.array(cleaned_ids))
merged.to_csv('merged_cleaned.csv', header=True, index=False)








import pandas as pd
import numpy as np
merged = pd.read_csv('merged_cleaned.csv')
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
 2629,
 1875,
 2094,
 2098,
 2288,
 2289,
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
print(len(merged.columns))
merged.drop(['activity_'+str(a) for a in unused_aids], axis=1, inplace=True)
print(len(merged.columns))
print(len(merged))
activity_columns = [c for c in merged.columns if 'activity' in c]
merged.dropna(subset=activity_columns, how='all', inplace=True)
print(merged.shape)
merged.to_csv('merged_cleaned_benchmarked.csv', header=True, index=False)

df = pd.read_csv('merged_features_clustered_ner_cleaned_extracted_benchmarked_cleaned.csv')
active_num = []
total_num = []
for aid in activity_columns:
    aid_number = int(aid.lstrip('activity_'))
    df_dummy = df[df['AID'] == aid_number]
    active_num.append(len(merged[merged[aid] == 1]))
    total_num.append(len(merged[aid].dropna()))
    if not pd.isna(df_dummy['active_num']).bool():
        print(aid_number, int(df_dummy['substance_num']), int(df_dummy['active_num']), len(merged[aid].dropna()), len(merged[merged[aid] == 1]))
    else:
        print(aid_number, int(df_dummy['substance_num']), len(merged[aid].dropna()), len(merged[merged[aid] == 1]))

# import matplotlib.pyplot as plt
# plt.hist(active_num, bins=range(0,500,25))

print(len(active_num) - np.sum(np.array(active_num)>=50))


df['recovered_substance_num'] = np.array(total_num)
df['recovered_active_num'] = np.array(active_num)

df.to_csv('merged_features_clustered_ner_cleaned_extracted_benchmarked_cleaned_counted.csv', header=True, index=False)


df = pd.read_csv('merged_features_clustered_ner_cleaned_extracted_benchmarked_cleaned_counted.csv')
print(len(df))
df = df[df['recovered_active_num'] >= 15]
print(len(df))
df.to_csv('merged_features_clustered_ner_cleaned_extracted_benchmarked_cleaned_counted_threshold.csv', header=True, index=False)

