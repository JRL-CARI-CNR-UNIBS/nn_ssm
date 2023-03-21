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
fig_name = str(dof)+"dof.png"
nn_name = "nn_ssm.pt"
dataset_name = "ssm_dataset.bin"
batch_size = 32
n_epochs = 500

# Get paths
PATH = os.path.dirname(os.path.abspath(__file__)) + "/data/"
dataset_path = PATH + dataset_name
nn_path = PATH + nn_name
fig_path = PATH + fig_name

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Load dataset
raw_data = np.fromfile(dataset_path, dtype='float')
length = raw_data.size

cols = dof + dof + 3 + 1  # q, dq, (x,y,z) of obstacle, scalings
rows = int(length/cols)

# from array to multi-dimensional-array
raw_data = raw_data.reshape(rows, cols)

# Set max scalings value
scalings = raw_data[:, -1]  # last column
scalings = np.where(scalings > max_scaling, max_scaling, scalings)

# Create training and validation datasets

# q, dq, (x,y,z) of obstacle (last column excluded)
input = torch.Tensor(raw_data[:, 0:-1])
target = torch.Tensor(scalings)

# random shuffle
indices = torch.randperm(input.size()[0])
input = input[indices]
target = target[indices]

dataset = torch.utils.data.TensorDataset(input, target)  # create your datset

train_size = int(rows*0.8)
val_size = rows-train_size

train_dataset = torch.utils.data.TensorDataset(
    input[0:train_size+1, :], target[0:train_size+1])  # create your train datset
val_dataset = torch.utils.data.TensorDataset(
    input[train_size+1:, :], target[train_size+1:])   # create your validation datset

if train_size == rows:
    val_dataset = train_dataset

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=True)

if load_net:
    NN = torch.load(nn_path)
else:
    NN = nn.Sequential(
        nn.Linear(dof+dof+3, 500),
        nn.ReLU(),
        nn.Linear(500, 1),
        nn.ReLU()).to(device)

print(NN)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
# criterion = torch.nn.L1Loss()

optimizer = optim.Adam(NN.parameters(), lr=0.001)
# optimizer = optim.SGD(NN.parameters(), lr=0.001, momentum=0.9)

# Train & Validation

# prepare for visualization
val_loss_over_epoches = []
train_loss_over_epoches = []

# fig, axs = plt.subplots(2)
# fig = plt.Figure()
# fig.set_size_inches(10, 10)
# ax0 = fig.add_subplot(1,2,1)
# ax1 = fig.add_subplot(1,2,2)

fig, axs = plt.subplots(2)
ax0 = axs[0]
ax1 = axs[1]
fig.set_size_inches(10,10)

for epoch in range(n_epochs):

    # Training phase
    print("----- TRAINING -----")

    NN.train()  # set training mode

    train_loss = 0.0
    batches_loss = 0.0

    for i, batch in enumerate(train_dataloader, 0):  # batches
        # get the inputs -> batch is a list of [inputs, targets]
        batch_input, batch_targets = batch
        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)

        batch_targets = batch_targets.unsqueeze(1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        predictions = NN(batch_input)

        # backpropagation
        loss = criterion(predictions, batch_targets)
        loss.backward()

        optimizer.step()

        # print current performance
        train_loss += (loss.item()*batch_targets.size()[0])

        batches_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, batches_loss/1000))
            batches_loss = 0.0

    # Validation phase
    print("----- VALIDATION -----")

    val_loss = 0.0

    with torch.no_grad():  # evaluation does not need gradients computation
        NN.eval()  # set evaluation mode

        # if epoch % 10 == 9:    # print the prediction error every 10 epoches
        ax1.clear()
        ax1.set(xlabel="Target", ylabel="Error",
                title="Prediction Error")
        ax1.grid(True)

        for i, batch in enumerate(val_dataloader, 0):
            # get the inputs -> batch is a list of [inputs, targets]
            batch_input, batch_targets = batch
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_targets = batch_targets.unsqueeze(1)

            predictions = NN(batch_input)
            loss = criterion(predictions, batch_targets)
            val_loss += (loss.item()*batch_targets.size()[0])

            # if epoch % 10 == 9:    # print the prediction error every 10 epoches
            ax1.plot(batch_targets.detach().cpu().numpy(), predictions.detach(
            ).cpu().numpy()-batch_targets.detach().cpu().numpy(), '.')

    # Plot figures
    train_epoch_loss = train_loss/train_size
    train_loss_over_epoches.append(train_epoch_loss)

    val_epoch_loss = val_loss/val_size
    val_loss_over_epoches.append(val_epoch_loss)

    ax0.clear()
    ax0.set(xlabel="Epoches", ylabel="Loss",
               title="Training and Validation Loss")
    ax0.grid(True)
    ax0.plot(val_loss_over_epoches)
    ax0.plot(train_loss_over_epoches)
    ax0.legend(["val","train"])

    plt.show(block=False)
    plt.draw()
    plt.pause(0.0001)

    # Save NN at each epoch
    torch.save(NN, nn_path)
    fig.savefig(fig_path, dpi=300)