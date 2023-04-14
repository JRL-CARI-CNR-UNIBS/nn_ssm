import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml 
from torch.nn.utils import parameters_to_vector

# Params
namespace = "ssm_nn"

PATH = os.path.dirname(os.path.abspath(__file__)) + "/data/"

NN = nn.Sequential(
    nn.Linear(15, 100),
    nn.Tanh(),
    nn.Linear(100, 100),
    nn.Tanh(),
    nn.Linear(100, 100),
    nn.Tanh(),
    nn.Linear(100, 13),
    nn.Sigmoid()
)
print(NN)

last_key = ""
nn_dict = {}
i = 0
n_inputs = 0
for layer in NN.children():

    if isinstance(layer, nn.Linear):
        if i == 0:
            n_inputs = layer.in_features
            nn_dict["inputs"] =  n_inputs
        
        n_neurons = layer.out_features
        weights = parameters_to_vector(layer.weight).detach().cpu().numpy().tolist()
        bias = parameters_to_vector(layer.bias).detach().cpu().numpy().tolist()
        last_key = "layer"+str(i)
        nn_dict[last_key] = {"neurons": n_neurons,"weights": weights, "bias": bias}
        
        i = i+1
    elif isinstance(layer, nn.ReLU):
        nn_dict[last_key].update({"activation": "relu"})
    elif isinstance(layer, nn.Tanh):
        nn_dict[last_key].update({"activation": "tanh"})
    elif isinstance(layer, nn.Sigmoid):
        nn_dict[last_key].update({"activation": "sigmoid"})
    else:
        raise Exception("Sorry, layer type not implemente yet")

dict = {namespace: nn_dict}

with open(PATH+"MODEL_test_speed.yaml", 'w') as f:
    yaml.dump(dict, f)   
