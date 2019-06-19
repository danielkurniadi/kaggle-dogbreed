import os
import sys
import glob

# data prep tools
import pandas as pd
import pickle
import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

__all__ = ['id_to_path', 'num_encoding_labels_file', 
    'split_from_datadir', 'split_path_label_pairs']

PATHS_COLNAME = 'paths'
LABELS_COLNAME = 'labels'

def id_to_path(datadir, labels_file, id_colname, outpath, sep=',', ext='.jpg'):
    # load labels from labels file
    labels_df = pd.read_csv(labels_file, sep=sep)
    labels_df[id_colname] =  labels_df[id_colname].apply(lambda idname: os.path.join(datadir, idname + ext))
    labels_df.rename(columns={id_colname: PATHS_COLNAME}, inplace=True)
    labels_df.to_csv(outpath, index=False)

def num_encoding_labels_file(labels_file, label_name, outpath, sep=","):
    # load labels from labels file
    labels_df = pd.read_csv(labels_file, sep=sep)
    label_col = labels_df[label_name]
    
    # encode to number 
    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(label_col)
    mapping =  dict(zip(lbl_encoder.classes_, lbl_encoder.transform(lbl_encoder.classes_)))
    labels_df[LABELS_COLNAME] = lbl_encoder.transform(label_col)

    # write to csv file
    labels_df.to_csv(outpath, index=False)
    
    # write to pickle
    outpath, ext = os.path.splitext(outpath)
    outdir = os.path.dirname(outpath)
    picklefile = os.path.join(outdir, "metadata.pkl")
    with open(picklefile, 'wb') as fp:
        pickle.dump(mapping, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    return labels_df, mapping

def _skf_path_labels(all_paths, all_labels, outdir, out_prefix="", n_splits=5):
    # stratify dataset
    skf = StratifiedKFold(n_splits=n_splits)
    for i, (train_idx, test_idx) in enumerate(skf.split(all_paths, all_labels)):
        X_train = all_paths[train_idx] # X_train is list of train data path 
        y_train = all_labels[train_idx] # y_train is list of label values

        # path for text of train path list
        train_prefix = "{}train_split_{}.txt".format(out_prefix, i)
        train_filepath = os.path.join(outdir, train_prefix) 

        with open(train_filepath, 'w') as fp:
            for filepath, label in zip(X_train, y_train):
                fp.write("{} {}\n".format(filepath, label))
        
        X_test = all_paths[test_idx] # X_test is list of train data path
        y_test = all_labels[test_idx] # y_test is list of train data path

        # path for text of test path list
        test_prefix = "{}val_split_{}.txt".format(out_prefix, i)
        test_filepath = os.path.join(outdir, test_prefix) 

        with open(test_filepath, 'w') as fp:
            for filepath, label in zip(X_test, y_test):
                fp.write("{} {}\n".format(filepath, label))

def split_from_datadir(datadir, outdir, n_splits=5, out_prefix=""):
    """Split data to k folds of train and test set using their paths.
    This will take data folder and generate k text files, each contains lists of data paths.
    Assuming data is put into folder based on their class/label.

    Arguments:
        - datadir (str): top directory of data folder.
        - outdir (str): path to output generated text files
        - n_splits (int): number of folds (k)
        - out_prefix (str): if provided, will prefix the text filename with it.
    """
    class_folders = [dirr for dirr in os.listdir(os.path.abspath(datadir)) if os.path.isdir(dirr)]
    labels = class_folders
    class_folders = [os.path.abspath(dirr) for dirr in class_folders]

    all_paths = []
    all_labels = []
    # concatenate all data paths
    for i, dirr in enumerate(class_folders):
        paths = glob.glob(os.path.join(dirr, '*'))
        labels = [class_folders[i]] * len(paths)
        all_paths.extend(paths)
        all_labels.extend(labels)
    
    # stratify
    _skf_path_labels(all_paths, all_labels, outdir)

def split_path_label_pairs(labels_file, outdir, sep=",", has_header=True, n_splits=5, out_prefix=""):
    """Split path-label pairs to k folds of train and test set.
    This will take labels file and generate k text files each contains lists of data paths.
    Labels file should contains file name (data) and label (2 columns) in each line, seperated by separator.

    Arguments:
        - labels_file (str): filename/path for labels file
        - outdir (str): specify dir for saving skf text files
        - sep (char): specify separator used in labels_file
        - has_header (bool): whether the 1st row in labels_file is a header
        - n_splits (int): number of splits (k)
        - out_prefix (str): if provided, will prefix the text filename with it
    """
    labels_df = pd.read_csv(labels_file, sep=sep)
    all_paths = labels_df[PATHS_COLNAME].to_numpy()
    all_labels = labels_df[LABELS_COLNAME].to_numpy()
    
    _skf_path_labels(all_paths, all_labels, outdir)

