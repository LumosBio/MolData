import deepchem as dc
import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
input_data = 'merged_cleaned_benchmarked_threshold_scaffold_split_stratified.csv'
input_columns = list(pd.read_csv(input_data).columns)
input_tasks = list(np.array(input_columns)[[True if 'activity' in c else False for c in input_columns]])
print(input_tasks)
split = 'specified'
featurizer = 'ECFP'


data_dir = input_data

# assign data and tasks
dataset_file = data_dir
tasks = input_tasks
valid_indices, test_indices = None, None
if split == 'specified':
    dummy_df = pd.read_csv(data_dir, low_memory=False)
    valid_indices = dummy_df.index[dummy_df['split'] == 'validation'].tolist()
    test_indices = dummy_df.index[dummy_df['split'] == 'test'].tolist()
print("About to load the dataset.")

# create featurizer, loader, transformers, and splitter
if featurizer == 'ECFP':
    featurizer = dc.feat.CircularFingerprint(size=1024, chiral=True)
elif featurizer == 'GraphConv':
    featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
loader = dc.data.CSVLoader(tasks=tasks, feature_field="smiles", featurizer=featurizer)
splitters = {
    'scaffold': dc.splits.ScaffoldSplitter(),
    'specified': dc.splits.SpecifiedSplitter(valid_indices=valid_indices, test_indices=test_indices)
}
splitter = splitters[split]


if not os.path.exists(dataset_file):
    print("Dataset not found")

print("About to featurize the dataset.")
dataset = loader.create_dataset([dataset_file], shard_size=8192)

print("About to transform data")
transformers = [dc.trans.BalancingTransformer(dataset=dataset)]
for transformer in transformers:
    dataset = transformer.transform(dataset)
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)
# dc.utils.data_utils.save_dataset_to_disk(data_save_dir, train=train_dataset, valid=valid_dataset, test=test_dataset, transformers=transformers)

# Extract the smiles for each split
train_smiles = np.array(train_dataset.ids)
valid_smiles = np.array(valid_dataset.ids)
test_smiles = np.array(test_dataset.ids)

train_features = np.array(train_dataset.X)
valid_features = np.array(valid_dataset.X)
test_features = np.array(test_dataset.X)

all_features = np.concatenate((train_features, valid_features, test_features))
print(all_features.shape)
# Save the smiles back into the CSV file
all_smiles = np.array(list(train_smiles) + list(valid_smiles) + list(test_smiles))
labels_smiles = np.array(['train'] * len(train_smiles) + ['validation'] * len(valid_smiles) + ['test'] * len(test_smiles))
smiles_df = pd.DataFrame(data=all_features, columns=['feature_' + str(i) for i in range(1024)])
smiles_df.insert(loc=0,column='smiles', value=all_smiles)
smiles_df.insert(loc=1,column='split', value=labels_smiles)
smiles_df.to_csv('smiles_ecfp.csv', header=True, index=False)


# Read the SMILES again for Tanimoto Coefficient calculations
smiles_df = pd.read_csv('smiles_ecfp.csv')
feature_columns = []
for c in smiles_df.columns:
    if 'feature' in c:
        feature_columns.append(c)


def largest_tanimoto_similarity(f1_bool, f2_bool):
    # f1 is one boolean numpy array, containing the molecular fingerprint for one molecule
    # f2 is (N-1)*M boolean numpy matrix, containing molecular fingerprints for all molecules except f1
    # Returns the largest Tanimoto Coefficient between f1 and the rest of the fingerprint (most similar)
    f1_bool = np.tile(f1_bool, (len(f2_bool), 1))
    # Overlap between "ones" from f1 and "ones" from the rest of the dataset
    overlap = np.sum(np.logical_and(f1_bool, f2_bool), axis=1)
    # Union between "ones" from f1 and "ones" from the rest of the dataset
    union = np.sum(np.logical_or(f1_bool, f2_bool), axis=1)
    return np.max(overlap/union)


tanimoto_scores = []
fingerprint_array = np.array(smiles_df[feature_columns].sample(n=200000, random_state=42), dtype=bool)
# fingerprint_array = np.array(smiles_df[feature_columns], dtype=bool)
for i in range(len(fingerprint_array)):
    indices = np.arange(len(fingerprint_array))
    dummy_score = largest_tanimoto_similarity(fingerprint_array[i, :], fingerprint_array[indices != i, :])
    tanimoto_scores.append(dummy_score)
    print(i, dummy_score)

np.save('results/tanimoto_scores', np.array(tanimoto_scores))


fontsize = 13
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
# plot the cumulative histogram
n, bins, patches = ax.hist(tanimoto_scores, 200, density=True, histtype='step',
                           cumulative=True)
# Find the percentage where tanimoto score is 0.5
index_tanimoto_7 = 1 - np.sum(np.array(tanimoto_scores) > 0.7) / len(tanimoto_scores)
index_tanimoto_5 = 1 - np.sum(np.array(tanimoto_scores) > 0.5) / len(tanimoto_scores)

# tidy up the figure
ax.grid(True)
plt.yticks([0,0.2,0.4,0.6,0.8,1], [0,20,40,60,80,100], fontsize=fontsize)
plt.xticks(fontsize=fontsize)
ax.set_ylabel('Cumulative % of Molecules', labelpad=0, fontsize=fontsize)
ax.set_xlabel('Largest Tanimoto Coefficient', labelpad=0, fontsize=fontsize)

# add one point in the 50 percentile
plt.annotate('(0.7,'+str(round(index_tanimoto_7*100, 2))+')',  # this is the text
             (0.70, index_tanimoto_7),  # these are the coordinates to position the label
             textcoords="offset points",  # how to position the text
             xytext=(0, 3),  # distance from text to points (x,y)
             ha='right',
             fontsize=fontsize)  # horizontal alignment can be left, right or center
ax.scatter([0.7], [index_tanimoto_7], c='black', s=7, zorder=3)
plt.annotate('(0.5,'+str(round(index_tanimoto_5*100, 2))+')',  # this is the text
             (0.50, index_tanimoto_5),  # these are the coordinates to position the label
             textcoords="offset points",  # how to position the text
             xytext=(0, 3),  # distance from text to points (x,y)
             ha='right',
             fontsize=fontsize)  # horizontal alignment can be left, right or center
ax.scatter([0.5], [index_tanimoto_5], c='black', s=7, zorder=3)
plt.tight_layout()
plt.savefig('results/cumulative_tanimoto.png', format='png', dpi=300)
plt.show()
