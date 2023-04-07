import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml 
from torch.nn.utils import parameters_to_vector

# Params
nn_name = "nn_ssm_complete.pt"

PATH = os.path.dirname(os.path.abspath(__file__)) + "/data/"

NN = torch.load(PATH+nn_name)

nn_dict = {}
i = 0
for layer in NN.children():
    if isinstance(layer, nn.Linear):
        weights = parameters_to_vector(layer.weight).detach().cpu().numpy().tolist()
        bias = parameters_to_vector(layer.bias).detach().cpu().numpy().tolist()
        nn_dict["layer"+str(i)] = {"weights": weights, "bias": bias}
        i = i+1

with open(PATH+"MODEL.yaml", 'w') as f:
    yaml.dump(nn_dict, f)  

# for name, param in NN.named_parameters():
#     print(name,param)
#     yaml_data = parameters_to_vector(param).detach().cpu().numpy()
#     with open(PATH+"MODEL.yaml", 'w') as f:
#         yaml.dump(yaml_data.tolist(), f)  

    # yaml_data = parameters_to_vector(NN.parameters()).detach().cpu().numpy()
    # with open(PATH+"MODEL.yaml", 'w') as f:
    #     yaml.dump(yaml_data.tolist(), f)  

