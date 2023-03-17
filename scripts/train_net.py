import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Params
dof = 6
load_net = False
max_scaling = 1000
nn_name = "nn_ssm.pt"
dataset_name = "ssm_dataset.bin"

# Get paths
PATH = os.path.dirname(os.path.abspath(__file__)) + "/scripts/data/"
dataset_path = PATH + dataset_name
nn_path = PATH + nn_name

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Load dataset
raw_data = np.fromfile(dataset_path, dtype='float') # array containing the data
length = raw_data.size

cols = dof + dof + 3 + 1  # q, dq, (x,y,z) of obstacle, scaling
rows = int(length/cols)

raw_data = raw_data.reshape(rows,cols) # from array to multi-dimensional-array

# Set max scaling value 
scaling = raw_data[:,cols] # last column
scaling = np.where[scaling>max_scaling,max_scaling,scaling]

# Define inputs and outputs
input  = torch.Tensor(raw_data[:,0:cols-1])  # q, dq, (x,y,z) of obstacle
output = torch.Tensor(scaling)

print(f"First two samples {input[0:1,:]}")

# TODO: fai dataset di train e di test

dataset = torch.utils.data.TensorDataset(input,output) # create your datset
dataloader = torch.utils.data.DataLoader(dataset,100) # create your dataloader
dataloader_plot = torch.utils.data.DataLoader(dataset,100000) # create your dataloader

if load_net:
   net=torch.load(nn_path)
else:
   net = nn.Sequential(
         nn.Linear(dof+3, 500),
         nn.ReLU(),
         nn.Linear(500, 100),
         nn.ReLU(),
         nn.Linear(100, 50),
         nn.ReLU(),
         nn.Linear(50, 10),
         nn.ReLU(),
         nn.Linear(10, 1)
         ).to(device)


# dropout

print(net)
print(net(input[1,:]))

criterion = torch.nn.L1Loss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)

ordine=2.0

for epoch in range(5000):  # loop over the dataset multiple times

   running_loss = 0.0
   for i, data in enumerate(dataloader, 0):
       # get the inputs; data is a list of [inputs, labels]
       inputs, distance = data

       # zero the parameter gradients
       optimizer.zero_grad()

       # forward + backward + optimize
       outputs = net(inputs)
#       loss = criterion(100*outputs/(distance+1), 100*distance/(distance+1))
       loss = criterion(0.1*outputs,0.1*distance)
       loss.backward()
       optimizer.step()

       # print statistics
       running_loss += loss.item()
       if i % 2000 == 1999:    # print every 2000 mini-batches
           print('[%d, %5d] loss: %.3f' %
                 (epoch + 1, i + 1, running_loss / 2000))
           running_loss = 0.0


   if epoch % 10 == 9:    # print every 2000 mini-batches
       for i, data in enumerate(dataloader_plot, 0):
           inputs, distance = data
           model_output=net(inputs)

           fig, axs = plt.subplots(2)
           fig.set_size_inches(10,10)

           axs[0].plot(distance.numpy(),model_output.detach().numpy(),'.')
           axs[0].set(xlabel="distance",ylabel="estimated distance",title="dof"+str(dof))
           axs[0].grid(True)
           axs[1].plot(distance.numpy(),model_output.detach().numpy()-distance.numpy(),'.')
           axs[1].set(xlabel="distance",ylabel="model-real",title="dof"+str(dof))
           axs[1].grid(True)
           #fig.savefig("dof3.png",dpi=300)
       plt.show()
       torch.save(net, nn_path)
