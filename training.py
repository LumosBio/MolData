import pandas as pd
import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models import GraphConvModel
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, recall_score, average_precision_score, precision_score
from deepchem.metrics.score_function import bedroc_score
import time
import os
from rdkit.Chem import MolFromSmiles, MolToSmiles
import shutil
import logging
import itertools
from typing import Any, Dict, List, Iterator, Optional, Sequence, Tuple
from deepchem.splits import Splitter
from deepchem.data import Dataset, DiskDataset


class MolDataSpecifiedSplitter(Splitter):
  """Split data in the fashion specified by user. Uses DeepChem's specified
  splitter but also takes as input the training splits.

  For some applications, you will already know how you'd like to split the
  dataset. In this splitter, you simplify specify `train_indices`, `valid_indices` and
  `test_indices` and the datapoints at those indices are pulled out of the
  dataset. Note that this is different from `IndexSplitter` which only splits
  based on the existing dataset ordering, while this `SpecifiedSplitter` can
  split on any specified ordering.
  """

  def __init__(self,
               train_indices: Optional[List[int]] = None,
               valid_indices: Optional[List[int]] = None,
               test_indices: Optional[List[int]] = None
               ):
    """
    Parameters
    -----------
    valid_indices: List[int]
      List of indices of samples in the valid set
    test_indices: List[int]
      List of indices of samples in the test set
    """
    self.train_indices = train_indices
    self.valid_indices = valid_indices
    self.test_indices = test_indices

  def split(self,
            dataset: Dataset,
            frac_train: float = 0.8,
            frac_valid: float = 0.1,
            frac_test: float = 0.1,
            seed: Optional[int] = None,
            log_every_n: Optional[int] = None
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits internal compounds into train/validation/test in designated order.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    frac_train: float, optional (default 0.8)
      Fraction of dataset put into training data.
    frac_valid: float, optional (default 0.1)
      Fraction of dataset put into validation data.
    frac_test: float, optional (default 0.1)
      Fraction of dataset put into test data.
    seed: int, optional (default None)
      Random seed to use.
    log_every_n: int, optional (default None)
      Log every n examples (not currently used).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
      A tuple of train indices, valid indices, and test indices.
      Each indices is a numpy array.
    """
    if self.train_indices is None:
      self.train_indices = []
    if self.valid_indices is None:
      self.valid_indices = []
    if self.test_indices is None:
      self.test_indices = []

    return (np.array(self.train_indices), np.array(self.valid_indices),
            np.array(self.test_indices))


def MolDataLoadData(data_dir, tasks, featurizer):
    dummy_df = pd.read_csv(data_dir, low_memory=False)
    train_indices = dummy_df.index[dummy_df['split'] == 'train'].tolist()
    valid_indices = dummy_df.index[dummy_df['split'] == 'validation'].tolist()
    test_indices = dummy_df.index[dummy_df['split'] == 'test'].tolist()
    print(len(dummy_df), len(train_indices) + len(valid_indices) + len(test_indices))
    print("About to load the dataset.")

    # create featurizer, loader, transformers, and splitter
    if featurizer == 'ECFP':
        featurizer = dc.feat.CircularFingerprint(size=1024, chiral=True)
    elif featurizer == 'GraphConv':
        featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
    loader = dc.data.CSVLoader(tasks=tasks, feature_field="smiles", featurizer=featurizer)
    splitters = {
        'specified': MolDataSpecifiedSplitter(train_indices=train_indices, valid_indices=valid_indices, test_indices=test_indices)
    }
    splitter = splitters['specified']

    if not os.path.exists(data_dir):
        print("Dataset not found")
    print("About to featurize the dataset.")
    dataset = loader.create_dataset([data_dir], shard_size=8192, data_dir='tmp/loader/')

    # Initialize transformers
    print("About to split data")
    untransformed_train_dataset, untransformed_valid_dataset, untransformed_test_dataset = \
        splitter.train_valid_test_split(dataset, train_dir='tmp/train_un/',
                                        valid_dir='tmp/valid_un/',
                                        test_dir='tmp/test_un/')
    print("About to transform data")
    transformers = [dc.trans.BalancingTransformer(dataset=dataset)]
    # Only transform the train dataset
    for transformer in transformers:
        train_dataset = transformer.transform(untransformed_train_dataset, out_dir='tmp/train/')

    shutil.rmtree('tmp/loader/', ignore_errors=True)
    return train_dataset, untransformed_valid_dataset, untransformed_test_dataset, transformers


#######################################################################
# Select operation mode, disease benchmarks or target benchmarks
data_type = 'disease'
# data_type = 'target'

# Select model and featurizer type
featurizer = 'GraphConv'
# featurizer = 'ECFP'

# Specify data directory
data_dir = 'data/all_molecular_data.csv'
map_df = pd.read_csv('data/aid_'+data_type+'_mapping.csv')
print(map_df.columns)

epochnb = 50
graph_conv_layers = [512, 512, 512]
dropout = 0.1
learning_rate = 0.0001
batch_size = 128
dense_layer_size = 1024
if data_type == 'disease':
    all_categories = ['all', 'cancer', 'nervous System', 'immune system', 'cardiovascular',
                          'toxicity', 'obesity', 'virus', 'diabetes', 'metabolic disorders', 'bacteria',
                          'parasite', 'epigenetics_genetics', 'pulmonary', 'infection', 'aging', 'fungal']
if data_type == 'target':
    all_categories = ['all_target', 'Membrane receptor', 'Enzyme (other)', 'Nuclear receptor',
           'Hydrolase', 'Protease', 'Transcription factor', 'Kinase',
           'Epigenetic regulator', 'Ion channel', 'Transferase', 'Oxidoreductase',
           'Transporter', 'NTPase', 'Phosphatase']

logging.basicConfig(level=logging.INFO)

for run_type in all_categories:
    np.random.seed(42)
    tf.compat.v1.set_random_seed(42)
    if run_type == 'all' or run_type == 'all_target':
        tasks = list(np.array(map_df['AID'])[[True if t > 0 else False for t in np.sum(np.array(map_df[all_categories[1:]]), axis=1)]])
    else:
        tasks = list(map_df[map_df[run_type] == 1]['AID'])
    # Select tasks based on the operation mode
    print(run_type, tasks)
    print(len(tasks))

    timestr = time.strftime("%m%d-%H%M")
    model_dir = 'built_models/moldata/'+featurizer+'/' + timestr + '/'
    if os.path.isdir(model_dir):
        timestr = timestr.split('-')[0] + '-' + timestr.split('-')[1][:2] + str(int(timestr.split('-')[1][2:])+60)
    os.makedirs(model_dir, exist_ok=True)

    # Load the data from the splits, transform only the train split
    train_dataset, untransformed_valid_dataset, untransformed_test_dataset, transformers = MolDataLoadData(data_dir=data_dir, tasks=tasks,featurizer=featurizer)
    training_data_len = len(train_dataset.y)


    metric = [
        dc.metrics.Metric(dc.metrics.accuracy_score, mode="classification", classification_handling_mode='threshold', threshold_value=0.5, n_tasks=len(tasks)),
        dc.metrics.Metric(dc.metrics.recall_score, mode="classification", classification_handling_mode='threshold', threshold_value=0.5, n_tasks=len(tasks)),
        dc.metrics.Metric(dc.metrics.precision_score, mode="classification", classification_handling_mode='threshold',
                          threshold_value=0.5, n_tasks=len(tasks)),
        dc.metrics.Metric(dc.metrics.roc_auc_score, mode="classification", n_tasks=len(tasks))]
    if featurizer == 'GraphConv':
        model = None
        model = GraphConvModel(
            len(tasks),
            batch_size=batch_size,
            mode='classification',
            number_atom_features=78,
            tensorboard=False,
            use_queue=True,
            graph_conv_layers=graph_conv_layers,
            dense_layer_size=dense_layer_size,
            dropout=dropout,
            learning_rate=learning_rate,
            model_dir=model_dir)

        for epoch_num in range(epochnb):
            loss = model.fit(train_dataset, nb_epoch=1, checkpoint_interval=2*(training_data_len // batch_size),
                             max_checkpoints_to_keep=1000)
            print(epoch_num)
    elif featurizer == 'ECFP':
        model = None
        model = dc.models.MultitaskClassifier(
            len(tasks),
            n_features=1024,
            layer_sizes=[dense_layer_size],
            dropouts=[dropout],
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_queue=False,
            model_dir=model_dir)
        loss = model.fit(train_dataset, nb_epoch=epochnb)
    results = model.evaluate(untransformed_test_dataset, metrics=metric, transformers=[], use_sample_weights=True, per_task_metrics=True)
    r = [list(results[0].values())]
    keys = list(results[0].keys())
    for i in range(len(results[1][keys[0]])):
        dummy = [results[1][k][i] for k in keys]
        r.append(dummy)
    r = np.array(r)
    print(r.shape)
    keys = [k + '_test' for k in keys]
    results_df = pd.DataFrame(data=np.array(r), columns=keys)
    results_df.insert(loc=0, column='AID', value=['all'] + tasks)
    results_df.insert(loc=len(results_df.columns), column='model_dir', value=[model_dir] * len(results_df))
    results_valid = model.evaluate(untransformed_valid_dataset, metrics=metric, transformers=[],
                                   use_sample_weights=True, per_task_metrics=True)
    r_valid = [list(results_valid[0].values())]
    keys_valid = list(results_valid[0].keys())
    for i in range(len(results_valid[1][keys_valid[0]])):
        dummy = [results_valid[1][k][i] for k in keys_valid]
        r_valid.append(dummy)
    r_valid = np.array(r_valid)
    print(r_valid.shape)
    keys_valid = [k + '_valid' for k in keys_valid]
    # results_df_valid = pd.DataFrame(data=np.array(r_valid), columns=keys_valid)
    for col in range(len(keys_valid)):
        results_df[keys_valid[col]] = np.array(r_valid)[:, col]
    # results_df_valid.insert(loc=0, column='AID', value=['all'] + input_tasks)

    results_df.to_csv('results/'+run_type+'_results.csv', header=True, index=False)
    shutil.rmtree('tmp/train_un/', ignore_errors=True)
    shutil.rmtree('tmp/valid_un/', ignore_errors=True)
    shutil.rmtree('tmp/test_un/', ignore_errors=True)
    shutil.rmtree('tmp/train/', ignore_errors=True)
    shutil.rmtree('tmp/valid/', ignore_errors=True)
    shutil.rmtree('tmp/test/', ignore_errors=True)


all_results = []
for c in all_categories:
    results_df = pd.read_csv('results/'+c+'_results.csv')
    results_df = results_df[results_df['AID'] == 'all']
    all_results.append(results_df[['accuracy_score_test', 'recall_score_test', 'precision_score_test',
                                   'roc_auc_score_test', 'accuracy_score_valid', 'recall_score_valid',
                                   'precision_score_valid', 'roc_auc_score_valid']].iloc[0])

all_results_df = pd.DataFrame(data=np.array(all_results), columns=['accuracy_score_test', 'recall_score_test', 'precision_score_test',
                                   'roc_auc_score_test', 'accuracy_score_valid', 'recall_score_valid',
                                   'precision_score_valid', 'roc_auc_score_valid'])
all_results_df.insert(loc=0, column='benchmark', value=all_categories)
all_results_df["accuracy_score_test"] = 100 * all_results_df["accuracy_score_test"]
all_results_df["recall_score_test"] = 100 * all_results_df["recall_score_test"]
all_results_df["precision_score_test"] = 100 * all_results_df["precision_score_test"]
all_results_df["accuracy_score_valid"] = 100 * all_results_df["accuracy_score_valid"]
all_results_df["recall_score_valid"] = 100 * all_results_df["recall_score_valid"]
all_results_df["precision_score_valid"] = 100 * all_results_df["precision_score_valid"]
all_results_df = all_results_df.round({'accuracy_score_test': 2, 'recall_score_test': 2, 'precision_score_test': 2, 'roc_auc_score_test': 4})
all_results_df = all_results_df.round({'accuracy_score_valid': 2, 'recall_score_valid': 2, 'precision_score_valid': 2, 'roc_auc_score_valid': 4})
all_results_df.to_csv('results/final_results_'+data_type+'.csv', header=True, index=False)


import pandas as pd
import numpy as np
all_disease_categories = ['all', 'cancer', 'nervous System', 'immune system', 'cardiovascular',
                          'toxicity', 'obesity', 'virus', 'diabetes', 'metabolic disorders', 'bacteria',
                          'parasite', 'epigenetics_genetics', 'pulmonary', 'infection', 'aging', 'fungal'] +\
                         ['all_target', 'Membrane receptor', 'Enzyme (other)', 'Nuclear receptor',
       'Hydrolase', 'Protease', 'Transcription factor', 'Kinase',
       'Epigenetic regulator', 'Ion channel', 'Transferase', 'Oxidoreductase',
       'Transporter', 'NTPase', 'Phosphatase']

writer = pd.ExcelWriter('results/detailed_results_combined.xlsx', engine = 'xlsxwriter')
for c in ['final_results_disease', 'final_results_target']:
    current_df = pd.read_csv('results/'+c+'.csv')
    if c == 'final_results_disease':
        c = 'benchmark_results_disease'
    if c == 'final_results_target':
        c = 'benchmark_results_target'
    current_df.insert(loc=1, column='run_type', value=c)
    current_df.to_excel(writer, sheet_name=c, index=False)

for c in all_disease_categories:
    current_df = pd.read_csv('results/'+c+'_results.csv')
    if c == 'all':
        c = 'all_disease'
    current_df.at[0, 'AID'] = c
    current_df.insert(loc=1, column='run_type', value=c)
    current_df.to_excel(writer, sheet_name=c, index=False)
    if c == 'all_disease':
        all_df = current_df
    else:
        all_df = all_df.merge(current_df, how='outer')

all_df = all_df.sort_values(by='AID', ignore_index=True)
all_df.to_csv('results/sorted_detailed_results_combined.csv', header=True, index=False)
all_df.to_excel(writer, sheet_name='everything', index=False)

writer.save()
writer.close()
