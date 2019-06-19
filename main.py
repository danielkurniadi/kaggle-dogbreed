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

from core import LOGS_DIR, DATA_DIR
from core.dataset import DogImageset
from core.models import Resnet50MO

DOGBREED_DIR = os.path.join(DATA_DIR, 'dogbreed/')
TRAIN_DIR = os.path.join(DOGBREED_DIR, 'raw/train/')
SPLITS_DIR = os.path.join(DOGBREED_DIR, 'splits/')

def main(ftrain_split, ftest_split):
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
    decay_factor = 10
    eval_freq = 5 # in epochs

    # data generator settings: dataset and dataloader
    train_dataset = DogImageset(ftrain_split, input_size,
        input_mean=input_mean, input_std=input_std)
    val_dataset = DogImageset(ftest_split, input_size,
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

    # experiment session
    for e in range(epochs):
        print("Epoch [{}/{}]".format(e+1, epochs))
        
        # training
        train(model, train_loader, criterion, optimizer, e+1)

        if (e+1)%5 == 0:
            learning_rate /= decay_factor
            update_learning_rate(optimizer, learning_rate)

        # validation
        validate(model, val_loader, criterion, optimizer)

        print("----------------------------------------")

    print("END: %s" % time.time())
    print('*' * 100)

def write_loss_to_logfile(file_suffix, epoch, iterr, loss):
    path = os.path.join(LOGS_DIR, "train_iters_loss_{}.csv".format(file_suffix))
    with open(path, 'a') as fp:
        fp.write("{}, {}, {}\n".format(epoch, iterr, loss))

def update_learning_rate(optimizer, learning_rate):
    for params_group in optimizer.params_groups:
        params_group['lr'] = learning_rate

def train(model, dataloader, criterion, optimizer, e):
    model.train()
    
    running_loss = 0.0
    running_correct = 0
    start = time.time()
    for i, (X, labels) in enumerate(dataloader):

        if torch.cuda.is_available():
            X = X.cuda()
            labels = labels.cuda()

        # feed forward & calculate loss
        outputs = model(X)
        loss = criterion(outputs, labels)

        # backpropagation and weights update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # eval acc
        _, preds = torch.max(outputs.data, 1)
        running_loss += loss.data
        running_correct += torch.sum(preds == labels.data)

        # print result
        elapsed = time.time() - start
        print("Iterations [{}/{}] | Train Loss: {: .4f}, | Elapsed time: {: .4f}".format(
            i+1, len(dataloader), loss.data, elapsed
        ))

        write_loss_to_logfile(model.__class__.__name__, e, i+1, loss.data)

    # print 1 epoch result
    print("Avg Train Loss: {: .4f} | Avg Train Acc: {: .4f}".format(
        running_loss/len(dataloader), running_correct/len(dataloader)
    ))

def validate(model, dataloader, criterion, optimizer):
    model.eval()
    
    running_loss = 0.0
    running_correct = 0

    for i, (X, labels) in enumerate(dataloader):

        if torch.cuda.is_available():
            X = torch.autograd.Variable(X.cuda())
            labels = torch.autograd.Variable(labels.cuda())
        
        # feed forward & calculate loss
        outputs = model(X)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs.data, 1)
        running_loss += loss.data[0]
        running_correct += torch.sum(preds == labels.data)

    # print 1 epoch result
    print("Avg Val Loss: {: .4f} | Avg Val Acc: {: .4f}".format(
        running_loss/len(dataloader), running_correct/len(dataloader)
    ))

if __name__ == '__main__':
    ftrain_split = os.path.join(SPLITS_DIR, 'train_split_0.txt')
    ftest_split = os.path.join(SPLITS_DIR, 'test_split_0.txt')
    main(ftrain_split, ftest_split)
