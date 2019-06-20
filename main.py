import os
import time
import shutil

import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

from core import LOGS_DIR, DATA_DIR, CHECKPOINT_DIR
from core.dataset import DogImageset
from core.models import Resnet50MO
from core.dataset.helpers import *

DOGBREED_DIR = os.path.join(DATA_DIR, 'dogbreed/')
TRAIN_DIR = os.path.join(DOGBREED_DIR, 'raw/train/')
SPLITS_DIR = os.path.join(DOGBREED_DIR, 'splits/')
LABELS_PATH = os.path.join(DOGBREED_DIR, 'labels.csv')
SPLIT_LABELS_PATH = os.path.join(SPLITS_DIR, 'labels.csv')

def main(ftrain_split, ftest_split, split):
    """Run model (resnet51) on dogbreed dataset
    """
    num_class = 120 # dogbreeds class
    model = Resnet50MO(num_class, checkpoint_path=None)

    # image settings
    # crop_size = model.crop_size
    # scale_size = model.scale_size
    input_size = model.input_size
    input_mean = model.input_mean
    input_std = model.input_std

    # hyperparams settings
    epochs = 20
    batch_size = 32 # mini-batch-size
    learning_rate = 0.01
    momentum = 0.5
    decay_factor = 0.35
    eval_freq = 3 # in epochs

    # data generator settings: dataset and dataloader
    train_dataset = DogImageset(ftrain_split, input_size,
        input_mean=input_mean, input_std=input_std)
    val_dataset = DogImageset(ftest_split, input_size,
        input_mean=input_mean, input_std=input_std)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Loss and backprop settings
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )

    # printing to widget
    print("----------------------------------------------------------------------------------------")
    print("TRAINING SESSION:")
    print("Model: %s" % model.__class__.__name__)
    print("Dataset: %s" % DogImageset.__name__)

    print("=======================================")
    print("HYPERPARAMS: ")
    print("Epochs: %d" % epochs)
    print("Batch-size: %d" % batch_size)
    print("Initial learning rate: %s" % learning_rate)

    print("========================================")
    print("SUMMARY")
    print("%s" % summary(model, (3, input_size, input_size)))

    print("BEGIN: %s" % time.time())

    best_acc = 0.0

    # experiment session
    for e in range(epochs):
        # training
        train(model, train_loader, criterion, optimizer, e+1)

        if (e+1)%eval_freq == 0:
            learning_rate *= decay_factor
            update_learning_rate(optimizer, learning_rate)

        # validation
        acc = validate(model, val_loader, criterion, optimizer, e+1)

        if (best_acc*1.05) < acc:
            best_acc = acc
            state = {
                'model_name': model.__class__.__name__,
                'split': split,
                'epoch': e+1,
                'state_dict': model.state_dict()
            }
            save_checkpoints(state)            

        print("----------------------------------------")

    print("END: %s" % time.time())
    print('*' * 100)

def write_loss_to_logfile(file_suffix, epoch, iterr, loss):
    path = os.path.join(LOGS_DIR, "train_iters_loss_{}.csv".format(file_suffix))
    with open(path, 'a') as fp:
        fp.write("{}, {}, {}\n".format(epoch, iterr, loss))

def save_checkpoints(state):
    filename = "checkpoints_{}_split{}_{}.pth".format(
        state['model_name'], state['split'], state['epoch']
    )
    savepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, savepath)

def update_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

def train(model, dataloader, criterion, optimizer, e):
    model.train()
    
    running_loss = 0.0
    running_acc = 0
    start = time.time()
    for i, (X, labels) in enumerate(dataloader):

        if torch.cuda.is_available():
            X = X.cuda()
            labels = labels.cuda()

        # feed forward & calculate loss
        outputs = model(X)
        loss = criterion(outputs, labels)
        running_loss += loss.data

        # backpropagation and weights update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # eval acc
        _, preds = torch.max(outputs.data, 1)
        acc = torch.mean((preds == labels.data).float())
        running_acc += acc 

        # print result
        elapsed = time.time() - start
        print("Epoch: [{:03d}] : Iterations [{:03d}/{}] | Train loss: {: .4f}, Train acc: {: .4f} | Elapsed time: {: .4f}".format(
            e, i+1, len(dataloader), loss.data, acc, elapsed
        ))

        write_loss_to_logfile(model.__class__.__name__, e, i+1, loss.data)

    # print 1 epoch result
    print("Epoch: [{:03d}] : Avg Train Loss: {: .4f} | Avg Train Acc: {: .4f}".format(
        e, running_loss/len(dataloader), running_acc/len(dataloader)
    ))

def validate(model, dataloader, criterion, optimizer, e):
    model.eval()
    
    running_loss = 0.0
    running_acc = 0

    for i, (X, labels) in enumerate(dataloader):

        if torch.cuda.is_available():
            X = torch.autograd.Variable(X.cuda())
            labels = torch.autograd.Variable(labels.cuda())
        
        # feed forward & calculate loss
        outputs = model(X)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs.data, 1)
        running_loss += loss.data
        acc = torch.mean((preds == labels.data).float())
        running_acc += acc

    # print 1 epoch result
    print("Epoch: [{:03d}] : Avg Val Loss: {: .4f} | Avg Val Acc: {: .4f}".format(
        e, running_loss/len(dataloader), running_acc/len(dataloader)
    ))

    return running_acc/len(dataloader)

def prepare_label_pairs():
    id_to_path(TRAIN_DIR, LABELS_PATH, 'id', SPLIT_LABELS_PATH)
    enclbl_df, mapping = num_encoding_labels_file(SPLIT_LABELS_PATH, "breed", SPLIT_LABELS_PATH)
    assert mapping
    split_path_label_pairs(SPLIT_LABELS_PATH, SPLITS_DIR)

if __name__ == '__main__':

    prepare_label_pairs()

    # prepare splits
    for k in range(0, 5 ):
        ftrain_split = os.path.join(SPLITS_DIR, 'train_split_{}.txt'.format(k))
        ftest_split = os.path.join(SPLITS_DIR, 'val_split_{}.txt'.format(k))
    
        main(ftrain_split, ftest_split, k)
