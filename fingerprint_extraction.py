import deepchem as dc
import numpy as np
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
input_data = 'merged_cleaned_benchmarked_threshold_scaffold_split_stratified.csv'
input_columns = list(pd.read_csv(input_data).columns)
input_tasks = list(np.array(input_columns)[[True if 'activity' in c else False for c in input_columns]])
print(input_tasks)
split = 'specified'
featurizer = 'ECFP'

# split = 'specified'
# featurizer = 'GraphConv'

# data_save_dir = 'cleaned_datasets/'+featurizer+'/'
data_dir = input_data

# os.makedirs(data_save_dir, exist_ok=True)

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
