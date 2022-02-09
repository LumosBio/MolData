# MolData - A Molecular Benchmark for Disease and Target Based Machine Learning

Deep learning’s automatic feature extraction has been a revolutionary addition to computational drug discovery, infusing both the capabilities of learning abstract features and discovering complex molecular patterns via learning from molecular data. Since biological and chemical knowledge is are necessary for overcoming the challenges of data curation, balancing, training, and evaluation, it is important for databases to contain meaningful information regarding the exact target and disease of each bioassay. The existing depositories such as PubChem or ChEMBL offer the screening data of for millions of molecules against a variety of cells and targets, however, their bioassays contain complex biological information descriptions which can hinder their usage by the machine learning community. In this work, a comprehensive disease and target-based dataset is collected from PubChem in order to facilitate and accelerate molecular machine learning for better drug discovery. MolData is one the largest efforts to date for democratizing the molecular machine learning, with roughly 170 million drug screening results from 1.4 million unique molecules assigned to specific diseases and targets. It also provides 30 unique categories of targets and diseases. Correlation analysis of the MolData bioassays unveil valuable information for drug repurposing for multiple diseases including cancer, metabolic disorders, and infectious diseases. Finally, we provide a benchmark of more than 30 models trained on each category using multitask learning. MolData aims to pave the way for computational drug discovery and accelerate the advancement of molecular artificial intelligence in a practical manner.

# Requirements
Requirements for training the models and running the benchmark:\
deepchem==2.5.0\
rdkit==2020.09.1\
tensorflow==2.5.0

Requirements for clustering the bioassay text descriptions:\
biobert-embedding==0.1.1\
transformers\
kneed\
python-docx

# How to Use
After cloning the repo, this repository can be used to perform training on the MolData dataset, or to create a molecular dataset from bioassays and their descriptions. To get bechmark result and to train model on the molecular data:\
1- Unzip the data within the data directory.\
2- Run training.py for training and evaluating a Graph Convolutional Neural Network, or a traditional ECFP-based fully connected network.\
If you plan to work with bioassays descriptions, move forward to "Preprocessing Bioassay Descriptions" section. In short you would need to preprocess the description, download molecular data, and clean and partition the molecular data.

# Data
The MolData dataset is can be accessed from the data directory after unzipping. all_molecular_data includes 1.4 million molecules, 600 columns of binary bioactivity labels, and the splits (training, validation, or test) the molecules belong to. aid_disease_mapping contains the mapping between bioassays and their related category of diseases, and aid_target_mapping contains the same for category of targets. To have accurate and comparable results, please use the provided split labels for training and evaluations.\
The data sources used for the creation of the MolData dataset gathered from the PubChem database are referenced within "data/data_reference_list.txt"

# Training on the Molecular Data
The training.py script offers simple training on the molecular data for all benchamarks, evaluates the trained models, and saves the results for each model. To start the training:\
1- Select the data type (disease or target) within the code.\
2 - Specify the featurizer (GraphConv or ECFP). GraphConv triggers training of a Graph Convolutional Neural Network, while ECFP trains a simple fully connected neural network.\
3 - Specify the training data directory (default is at data/ where you unzip the main dataset).\
4 - Start the training.

Training happens on a transformed training set to overcome imbalance, where positive data point have higher weights than the negative data points for the loss function. However, evaluation is done on untransformed validation and test sets, to not allow the transformed weights to affect the metric calculation outcomes and to allow missing values to not be counted toward the metric calculations.

# Preprocessing Bioassay Descriptions (optional)
Bioassays descriptions and summaries are downloaded from PuChem as text files for 9 different sources. The scripts follow this order:\
1- Preprocessing.py: Cleans the descriptions and extracts useful information from them using pre-defined rules.\
2- Clustering.py: Used BioBERT to extract features from the cleaned descriptions and titles, then used KMeans to cluster them. The cluster number are only used as recommendation during taggign each bioassay.\
3- Ner.py: Uses  a model trained for disease entity recognition to find all disease related words within the description. These words do not have an effect on the tagging, since the detected words were too broad.\
4- Manual_tag.py: After a human expert highlights the disease and target related words in all descriptions in a word files, these highlighted words are read and used for finding disease and target tags for each bioassay.

# Preprocessing Molecular Data (optional)
After the assays are found and tagged, the molecular data for each assay is downloaded from PubChem using PubChem's bulk download interface. The scripts regarding this section follow this order:\
1- Molecular_data_cleaning.py: Makes SMILES canon, cleans duplicate SMILES, adds binary labels to SMILES.\
2- Fingerprint_extraction.py: Extracts ECFP4 fingerprints from the data, then used Tanimoto Coefficient to calculate the diversity within the dataset.\
3- Correlation.py: Find linear correlation between the labels of all datasets (bioassays), this can be a starting step for drug repurposing.\
4- Splitting.py: Splits the molecular data to train, validation, and test splits using the molecular scaffolds.
