import numpy as np
import pandas as pd

from typing import Any, Dict, List, Iterator, Optional, Sequence, Tuple



from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles




def split(smiles,
          frac_train: float = 0.8,
          frac_valid: float = 0.1,
          frac_test: float = 0.1,
          seed: Optional[int] = None,
          log_every_n: Optional[int] = 1000
          ) -> Tuple[List[int], List[int], List[int]]:
    """
    Splits internal compounds into train/validation/test by scaffold.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    seed: int, optional (default None)
      Random seed to use.
    log_every_n: int, optional (default 1000)
      Controls the logger by dictating how often logger outputs
      will be produced.

    Returns
    -------
    Tuple[List[int], List[int], List[int]]
      A tuple of train indices, valid indices, and test indices.
      Each indices is a list of integers.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    scaffold_sets = generate_scaffolds(smiles)

    train_cutoff = frac_train * len(smiles)
    valid_cutoff = (frac_train + frac_valid) * len(smiles)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    # logger.info("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def generate_scaffolds(smiles_list,
                       log_every_n: int = 1000) -> List[List[int]]:
    """Returns all scaffolds from the dataset.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    log_every_n: int, optional (default 1000)
      Controls the logger by dictating how often logger outputs
      will be produced.

    Returns
    -------
    scaffold_sets: List[List[int]]
      List of indices of each scaffold in the dataset.
    """
    scaffolds = {}
    data_len = len(smiles_list)

    # logger.info("About to generate scaffolds")
    for ind, smiles in enumerate(smiles_list):
        # if ind % log_every_n == 0:
            # logger.info("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles), includeChirality=True)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets

input_data = 'data/all_molecular_data.csv'
smiles = np.array(pd.read_csv(input_data)['smiles'])
scaffold_list = []
counter = 0
for s in smiles:
    try:
        scaffold_list.append(MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s), includeChirality=True))
    except:
        scaffold_list.append(s)
        print(s, counter)
    counter += 1
df = pd.read_csv(input_data)
df['scaffold'] = np.array(scaffold_list)

df.dropna(subset=['smiles'], inplace=True)
print(len(df))
df.to_csv('merged_cleaned_benchmarked_threshold_scaffold.csv', header=True, index=False)

scaffold_list = np.array(df['scaffold'])
smiles = np.array(df['smiles'])

scaffolds = {}
for ind, scaffold in enumerate(scaffold_list):
    if scaffold not in scaffolds:
        scaffolds[scaffold] = [ind]
    else:
        scaffolds[scaffold].append(ind)

scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
scaffold_sets = [
    scaffold_set for (scaffold, scaffold_set) in sorted(
        scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
]

frac_train = 0.8
frac_valid= 0.1
frac_test = 0.1
train_cutoff = frac_train * len(smiles)
valid_cutoff = (frac_train + frac_valid) * len(smiles)
train_inds = []
valid_inds = []
test_inds = []

for scaffold_set in scaffold_sets:
    if len(train_inds) + len(scaffold_set) > train_cutoff:
        if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
            test_inds.extend(scaffold_set)
        else:
            valid_inds.extend(scaffold_set)
    else:
        train_inds.extend(scaffold_set)

split_list = np.empty(len(smiles), dtype=object)
for i in train_inds:
    split_list[i] = 'train'
for i in valid_inds:
    split_list[i] = 'validation'
for i in test_inds:
    split_list[i] = 'test'

print(len(split_list))
df['split'] = split_list
df.to_csv('merged_cleaned_benchmarked_threshold_scaffold_split.csv', header=True, index=False)

# df = pd.read_csv('merged_cleaned_benchmarked_threshold_scaffold_split.csv')
df = pd.read_csv('merged_cleaned_benchmarked_threshold_scaffold_split_stratified.csv')

activity_columns = [c for c in df.columns if 'activity' in c]
train_dummy = df[df['split'] == 'train']
valid_dummy = df[df['split'] == 'validation']
test_dummy = df[df['split'] == 'test']

problem_aids = []
for a in activity_columns:
    train_sum = int(np.sum(train_dummy[a]))
    valid_sum = int(np.sum(valid_dummy[a]))
    test_sum = int(np.sum(test_dummy[a]))
    if train_sum < 1 or valid_sum < 1 or test_sum <1:
        print(a, int(np.sum(train_dummy[a])), int(np.sum(valid_dummy[a])), int(np.sum(test_dummy[a])))
        problem_aids.append(a)

for a in activity_columns:
    train_sum = int(np.sum(list(train_dummy[a]==0)))
    valid_sum = int(np.sum(list(valid_dummy[a]==0)))
    test_sum = int(np.sum(list(test_dummy[a]==0)))
    if train_sum < 1 or valid_sum < 1 or test_sum <1:
        print(a, train_sum, valid_sum, test_sum)
        problem_aids.append(a)

for a in activity_columns:
    train_sum = int(np.sum(list(train_dummy[a]==0)))
    valid_sum = int(np.sum(list(valid_dummy[a]==0)))
    test_sum = int(np.sum(list(test_dummy[a]==0)))
    print(a, train_sum, int(np.sum(train_dummy[a])), valid_sum, int(np.sum(valid_dummy[a])), test_sum, int(np.sum(test_dummy[a])))

import random
extra_valid_ind = []
extra_test_ind = []
for a in problem_aids:
    dummy_df = df[df[a] == 1]
    aid_inds = dummy_df.index[dummy_df['split'] == 'train'].tolist()
    d1,d2 = random.sample(aid_inds, 2)
    extra_valid_ind.append(d1)
    extra_test_ind.append(d2)

for i in extra_valid_ind:
    df.at[i, 'split'] = 'validation'

for i in extra_test_ind:
    df.at[i, 'split'] = 'test'

train_dummy = df[df['split'] == 'train']
valid_dummy = df[df['split'] == 'validation']
test_dummy = df[df['split'] == 'test']

for a in activity_columns:
    train_sum = int(np.sum(train_dummy[a]))
    valid_sum = int(np.sum(valid_dummy[a]))
    test_sum = int(np.sum(test_dummy[a]))
    if train_sum < 1 or valid_sum < 1 or test_sum <1:
        print(a, int(np.sum(train_dummy[a])), int(np.sum(valid_dummy[a])), int(np.sum(test_dummy[a])))
df.to_csv('merged_cleaned_benchmarked_threshold_scaffold_split_stratified.csv', header=True, index=False)

df = pd.read_csv('merged_cleaned_benchmarked_threshold_scaffold_split_stratified.csv')
mini = df[:20000]
mini.to_csv('mini.csv', header=True, index=False)
