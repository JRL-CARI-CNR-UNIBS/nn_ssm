import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data.dataloader import default_collate

# Params
dof = 6
load_net = False
max_scaling = 1000
nn_name = "nn_ssm.pt"
dataset_name = "ssm_dataset.bin"
batch_size = 32
n_epochs = 1000

# Get paths
PATH = os.path.dirname(os.path.abspath(__file__)) + "/scripts/data/"
dataset_path = PATH + dataset_name
nn_path = PATH + nn_name

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Load dataset
raw_data = np.fromfile(dataset_path, dtype='float') # array containing the data
length = raw_data.size

cols = dof + dof + 3 + 1  # q, dq, (x,y,z) of obstacle, scalings
rows = int(length/cols)

raw_data = raw_data.reshape(rows,cols) # from array to multi-dimensional-array

# Set max scalings value 
scalings = raw_data[:,-1] # last column
scalings = np.where[scalings>max_scaling,max_scaling,scalings]

# Define inputs and outputs
input  = torch.Tensor(raw_data[:,0:-1])  # q, dq, (x,y,z) of obstacle (last column excluded)
output = torch.Tensor(scalings)

print(f"First two samples {input[0:2,:]}")

# Create trainingg and validation datasets
dataset = torch.utils.data.TensorDataset(input,output) # create your datset
dataset = dataset.shuffle(buffer_size=rows,seed=datetime.now())

train_size = int(rows*0.8)
test_size = int(rows*0.2)

train_dataset = dataset.take(train_size).to(device)    
test_dataset = dataset.skip(train_size).take(test_size).to(device) 

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: default_collate(x).to(device))
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: default_collate(x).to(device))    

if load_net:
   NN=torch.load(nn_path)
else:
   NN = nn.Sequential(
         nn.Linear(dof+dof+3, 100),
         nn.ReLU(),
         nn.Linear(100, 1),
         nn.ReLU()).to(device)

print(NN)
print(NN(input[1,:]))

# Define loss function and optimizers

#criterion = torch.nn.L1Loss()
criterion = torch.nn.MSELoss()

#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(NN.parameters(), lr=0.001)

# Train
train_loss = []
test_loss = []

for epoch in range(n_epochs):
   NN.train() # set training mode
   running_loss = 0.0
   for i, data in enumerate(train_dataloader, 0):  #batches
       inputs, scaling = data.to(device) #get the inputs; data is a list of [inputs, labels]

       # forward + backward + optimize
       optimizer.zero_grad() # zero the parameter gradients
       outputs = NN(inputs)
       loss = criterion(outputs,scaling)
       loss.backward()
       optimizer.step()

       # print statistics
       train_loss.append(loss.item())
       running_loss += loss.item()
       if i % 1000 == 999:    # print every 2000 mini-batches
           print('[%d, %5d] loss: %.3f' %
                 (epoch + 1, i + 1, running_loss / 1000))
           running_loss = 0.0

   if epoch % 10 == 9:  # print every 10 epoches
        with torch.no_grad:
            NN.eval() # set test mode
            for i, data in enumerate(test_dataloader, 0):
                inputs, scaling = data.to(device)
                test_output=NN(inputs)
                loss = criterion(test_output,scaling)
                test_loss.append(loss.item())
                
            plt.figure(figsize=(10,5))
            plt.title("Training and Test Loss")
            plt.plot(test_loss,label="val")
            plt.plot(train_loss,label="train")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

        torch.save(NN, nn_path)