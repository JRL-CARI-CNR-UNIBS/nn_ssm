import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Params
dof = 6
max_scaling = 100
fig_name = str(dof)+"dof.png"
nn_name = "nn_ssm.pt"
dataset_name = "ssm_dataset_10k.bin"

loss_fcn = ""
lr = 0.001

# Get paths
PATH = os.path.dirname(os.path.abspath(__file__)) + "/data/"
nn_path = PATH + nn_name
dataset_path = PATH + dataset_name

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

if device == "cuda":
    print(f"=> {torch.cuda.get_device_name(0)}")

# Load dataset
raw_data = np.fromfile(dataset_path, dtype='float')
length = raw_data.size

# speed, distance, q, dq, (x,y,z) of obstacle, scalings
cols = 1 + 1 + dof + dof + 3 + 1
rows = int(length/cols)

# from array to multi-dimensional-array
raw_data = raw_data.reshape(rows, cols)

# Transform scalings values
scalings = raw_data[:, -1]  # last column
scalings = np.where(scalings > max_scaling, 0.0, (1.0 /
                    scalings))   # scaling between 0.0 and 1.0

# Create training and validation datasets

# q, dq, (x,y,z) of obstacle (firs, second, last columns excluded)
input = torch.Tensor(raw_data[:, 0:-1]).reshape(rows, cols-1)
target = torch.Tensor(scalings).reshape(rows, 1)

# random shuffle
indices = torch.randperm(input.size()[0])
input = input[indices]
target = target[indices]

dataset = torch.utils.data.TensorDataset(
    input[0:int(0.8*rows),:], target[0:int(0.8*rows)])  # create your train datset

dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=100, shuffle=True, drop_last=True)

NN = torch.load(nn_path)

print(NN)

# Define loss function and optimizer

if loss_fcn == "L1":
    criterion = torch.nn.L1Loss(reduction='sum')
else:
    criterion = torch.nn.MSELoss()

optimizer = optim.Adam(NN.parameters(), lr=lr, amsgrad=True)
# optimizer = optim.SGD(NN.parameters(), lr=lr, momentum=0.7)

# Train & Validation

# prepare for visualization
plt.rcParams["figure.figsize"] = [10, 10]
ax = plt.GridSpec(1, 2)
ax.update(wspace=0.5, hspace=0.5)

ax0 = plt.subplot(ax[0, 0])
ax1 = plt.subplot(ax[0, 1])

NN = NN.train()  # set training mode

batches_loss = 0.0

train_targets = []
train_output = []
train_predictions_errors = []

for i, batch in enumerate(dataloader, 0):  # batches
    # get the inputs -> batch is a list of [inputs, targets]
    full_batch_input, batch_targets = batch
    batch_input = full_batch_input[:, 2:]

    batch_input = batch_input.to(device)
    batch_targets = batch_targets.to(device)

    with torch.no_grad():  # evaluation does not need gradients computation
        # forward
        predictions = NN(batch_input)

        # backpropagation
        loss = criterion(predictions, batch_targets)

    batches_loss = loss.item()
    print('[%5d] train batch loss: %.8f' %
        (i + 1, batches_loss))

    train_output.extend(predictions.detach().cpu().numpy())
    train_targets.extend(batch_targets.detach().cpu().numpy())
    train_predictions_errors.extend(batch_targets.detach().cpu().numpy()-predictions.detach().cpu(
    ).numpy())


    ax0.clear()
    ax0.set(xlabel="Target", ylabel="Error",
            title="Prediction Error")
    ax0.grid(True)
    ax0.plot(train_targets, train_predictions_errors, '.', color='orange')

    ax1.clear()
    ax1.set(xlabel="Target", ylabel="Prediction",
            title="Target-Prediction")
    ax1.grid(True)
    ax1.plot(train_targets, train_output, '.', color='orange')

    plt.show(block=False)
    plt.draw()
    plt.pause(0.0001)

plt.show()
