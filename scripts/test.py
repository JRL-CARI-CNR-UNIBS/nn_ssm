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
dataset_name = "ssm_dataset.bin"
n_epochs = 1000

n_inputs = 2
batch_size = 4

# Get paths
PATH = os.path.dirname(os.path.abspath(__file__)) + "/data/"
dataset_path = PATH + dataset_name

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

# Transform scalings values
scalings = raw_data[:, -1]  # last column

scalings = np.where(scalings > max_scaling, 0.0, (1.0 /
                    scalings))   # scaling between 0.0 and 1.0
# scalings = np.where(scalings > max_scaling, max_scaling, scalings)

# Create training and validation datasets

# q, dq, (x,y,z) of obstacle (last column excluded)
input = torch.Tensor(raw_data[:, 0:-1]).reshape(-1,cols-1)
target = torch.Tensor(scalings).reshape(-1,1)

# random shuffle
indices = torch.randperm(input.size()[0])
input = (input[indices])
target = (target[indices])

input = input[0:n_inputs, :]
target = target[0:n_inputs]

train_size = int(rows)
val_size = train_size

train_dataset = torch.utils.data.TensorDataset(
    input[0:train_size, :], target[0:train_size])  # create your train datset
val_dataset = train_dataset

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = train_dataloader

NN = nn.Sequential(
    nn.Linear(dof+dof+3, 100),
    nn.ReLU(),
    nn.Linear(100,1),
    nn.Sigmoid(),
).to(device)

print(NN)

# Define loss function and optimizer

criterion = torch.nn.L1Loss(reduction='sum')
#criterion = torch.nn.MSELoss()

optimizer = optim.Adam(NN.parameters(), lr=0.001)
#optimizer = optim.SGD(NN.parameters(), lr=0.001, momentum=0.7)

# Train & Validation

# prepare for visualization
val_loss_over_epoches = []
train_loss_over_epoches = []

fig, axs = plt.subplots(3)
ax0 = axs[0]
ax1 = axs[1]
ax2 = axs[2]
fig.set_size_inches(10, 10)

for epoch in range(n_epochs):

    # Training phase
    print("----- TRAINING -----")

    NN.train()  # set training mode

    train_loss = 0.0

    for i, batch in enumerate(train_dataloader, 0):  # batches
        # get the inputs -> batch is a list of [inputs, targets]
        batch_input, batch_targets = batch
        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)

        #batch_targets = batch_targets.unsqueeze(1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        predictions = NN(batch_input)

        # backpropagation
        loss = criterion(predictions, batch_targets)
        loss.backward()

        optimizer.step()

        # print current performance
        #train_loss += (loss.item()*batch_targets.size()[0])
        train_loss += loss.item()

        print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, loss.item()))

    # Validation phase

    val_loss = 0.0
    with torch.no_grad():  # evaluation does not need gradients computation
        NN.eval()  # set evaluation mode

        val_targets = []
        val_output = []
        predictions_errors = []
        for i, batch in enumerate(val_dataloader, 0):
            # get the inputs -> batch is a list of [inputs, targets]
            batch_input, batch_targets = batch
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            #batch_targets = batch_targets.unsqueeze(1)

            predictions = NN(batch_input)
            loss = criterion(predictions, batch_targets)
            # val_loss += (loss.item()*batch_targets.size()[0])
            val_loss += loss.item()

            val_output.extend(predictions.detach().cpu().numpy())
            val_targets.extend(batch_targets.detach().cpu().numpy())
            predictions_errors.extend(batch_targets.detach().cpu().numpy()-predictions.detach().cpu(
            ).numpy())

    # Plot figures
    train_epoch_loss = train_loss/(epoch+1)
    train_loss_over_epoches.append(train_epoch_loss)

    val_epoch_loss = val_loss/(epoch+1)
    val_loss_over_epoches.append(val_epoch_loss)

    ax0.clear()
    ax0.set(xlabel="Epoches", ylabel="Loss",
            title="Training and Validation Loss")
    ax0.grid(True)
    ax0.plot(val_loss_over_epoches)
    ax0.plot(train_loss_over_epoches)
    ax0.legend(["val", "train"])

    ax1.clear()
    ax1.set(xlabel="Target", ylabel="Error",
            title="Prediction Error")
    ax1.grid(True)
    ax1.plot(val_targets, predictions_errors, '.')

    ax2.clear()
    ax2.set(xlabel="Target", ylabel="Prediction",
            title="Target-Prediction")
    ax2.grid(True)
    ax2.plot(val_targets, val_output, '.')

    plt.show(block=False)
    plt.draw()
    plt.pause(0.0001)
