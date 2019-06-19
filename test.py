import os
import sys
import glob
import time
import shutil

# data prep tools
import pandas as pd
import pickle
import sklearn

import pretrainedmodels

import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

from core import DATA_DIR
from core.dataset import DogImageset
from core.models import Resnet50MO

from core.dataset.helpers import *


DOGBREED_DIR = os.path.join(DATA_DIR, 'dogbreed/')
TRAIN_DIR = os.path.join(DOGBREED_DIR, 'raw/train/')
SPLITS_DIR = os.path.join(DOGBREED_DIR, 'splits/')
ORI_LABELS_PATH = os.path.join(DOGBREED_DIR, 'labels.csv')
LABELS_PATH = os.path.join(DOGBREED_DIR, 'labels_en.csv')

def run_test(filepath):
    """Run model (resnet51) on dogbreed dataset
    """
    num_class = 120 # dogbreeds class
    model = Resnet50MO(num_class, checkpoint_path=None)

    # image settings
    crop_size = model.input_size
    scale_size = model.input_size
    input_size = model.input_size
    input_mean = model.input_mean
    input_std = model.input_std

    # hyperparams settings
    epochs = 1
    batch_size = 32 # mini-batch-size
    learning_rate = 0.01
    momentum = 0.5
    decay_factor = 10
    eval_freq = 5 # in epochs

    # data generator settings: dataset and dataloader
    train_dataset = DogImageset(filepath, input_size,
        input_mean=input_mean, input_std=input_std)
    val_dataset = DogImageset(filepath, input_size,
        input_mean=input_mean, input_std=input_std)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Loss and backprop settings
    # model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )

    run_model_train_test(model, train_loader, criterion, optimizer)

def run_model_train_test(model, dataloader, criterion, optimizer):
    X, labels = iter(dataloader).next()

    if torch.cuda.is_available():
        X = X.cuda()
        labels = labels.cuda()
    
    ################## DEBUG: INPUT ####################
    print("INPUT ---------------------------------------")
    print("X ({}):". format(X.shape))
    print(X)
    ####################################################
    print("labels ({}):".format(labels.shape))
    print(labels)
    ####################################################

    ################## DEBUG: FORWD ####################
    print("FORWD ---------------------------------------")
    outputs = model(X)
    print("Outputs ({}):".format(outputs.shape))
    print(outputs)
    ####################################################

    ################## DEBUG: BKPRP ####################
    print("BKPRP ---------------------------------------")
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print("Loss ({}):".format(loss.shape))
    print(loss)
    ####################################################

def prepare_label_pairs():
    id_to_path(TRAIN_DIR, ORI_LABELS_PATH, 'id', LABELS_PATH)
    enclbl_df, mapping = num_encoding_labels_file(LABELS_PATH, "breed", LABELS_PATH)
    assert mapping
    split_path_label_pairs(LABELS_PATH, SPLITS_DIR)

if __name__ == "__main__":
    
    prepare_label_pairs()
    filepath = os.path.join(SPLITS_DIR, 'train_split_0.txt')

    # run tests
    run_test(filepath)
