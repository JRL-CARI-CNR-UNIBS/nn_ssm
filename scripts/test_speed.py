import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import statistics 

# Params
dof = 6
load_net = False
max_scaling = 1000
fig_name = str(dof)+"dof.png"
nn_name = "1k_nn_ssm_complete.pt"
dataset_name = "ssm_dataset_1k.bin"
batch_size = 100

# Get paths
PATH = os.path.dirname(os.path.abspath(__file__)) + "/data/"
nn_path_shared = PATH + nn_name
dataset_path = PATH + dataset_name

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

if device == "cuda":
    print(f"=> {torch.cuda.get_device_name(0)}")

# Load dataset
raw_data = np.fromfile(dataset_path, dtype='float')
length = raw_data.size

# q, dq, (x,y,z) of obstacle, speed, distance, scalings
cols = dof + dof + 3 + 1 + 1 + 1
rows = int(length/cols)

# from array to multi-dimensional-array
raw_data = raw_data.reshape(rows, cols)

# Transform scalings values
scalings = raw_data[:, -1]  # last column
scalings = np.where(scalings > max_scaling, 0.0, (1.0 /
                    scalings))   # scaling between 0.0 and 1.0

# Create training and validation datasets

# q, dq, (x,y,z) of obstacle (first, second, last columns excluded)
input = torch.Tensor(raw_data[:, 0:-3]).reshape(rows, cols-3)
scalings_tensor = torch.Tensor(scalings).reshape(rows, 1)
speed_tensor = torch.Tensor(raw_data[:,-3]).reshape(rows, 1)
distance_tensor = torch.Tensor(raw_data[:,-2]).reshape(rows, 1)

target = torch.cat((speed_tensor,distance_tensor,scalings_tensor), -1)

# random shuffle
indices = torch.randperm(input.size()[0])
input = input[indices]
target = target[indices]

dataset = torch.utils.data.TensorDataset(input, target)  # create your datset
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

NN = torch.load(nn_path_shared)
NN = NN.eval()
print(NN)

time_vector = []

with torch.no_grad(): 
    for i, batch in enumerate(dataloader, 0):  # batches
        batch_input, batch_targets = batch

        start = time.time()
        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)
        
        predictions = NN(batch_input)
        end = time.time()

        time_required = (end-start)/len(batch_input)
        time_vector.append(time_required)

mean = statistics.mean(time_vector)
std_dev = statistics.stdev(time_vector)

print(f"mean: {mean}, std dev: {std_dev}")