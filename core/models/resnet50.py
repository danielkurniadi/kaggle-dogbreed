from collections import OrderedDict

import torch
from torch import nn
from torch.nn.init import normal, constant

import pretrainedmodels

class Resnet50MO(nn.Module):
    """Resnet 50 model and that's it.

    Arguments:
        - num_class (int): number of classification class.
        - checkpoint_path (str): If specified, will use the pretrained weight file (.pth) in the given path.
    """
    def __init__(self, num_class, checkpoint_path=None):

        super(Resnet50MO, self).__init__()
        self.num_class = num_class

        # image settings
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        # base model resnet50
        self.base_model = pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)

        # last fully-connected linear module settings
        self.base_model.last_layer_name = 'last_linear'
        num_feats = self.base_model.last_linear.in_features
        self.base_model.last_linear = torch.nn.Linear(num_feats, num_class)

    def forward(self, X):
        base_out = self.base_model(X)
        return base_out

