import numpy as np
import torch
from torch import nn

def create_model():
    # your code here
    # return model instance (None is just a placeholder)
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 16),
        nn.ReLU(),
        nn.Linear(16, 10)
    return model

def count_parameters(model):
    # your code here
    # return integer number (None is just a placeholder)
    sum_ = 0
    for w in model.parameters():
      w = w.shape
      if len(w) == 2:
        sum_ += w[0]*w[1]
      else:
        sum_ += w[0]
    return sum_

