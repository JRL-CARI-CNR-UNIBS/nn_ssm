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
dataset_name = "test_dataset.bin"
n_epochs = 5000
loss_fcn = ""

n_inputs = 1000
batch_size = 128
lr = 0.001  # 0.001

# Get paths
PATH = os.path.dirname(os.path.abspath(__file__)) + "/data/"
dataset_path = PATH + dataset_name
nn_path = PATH + nn_name
fig_path = PATH + fig_name

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

raw_data = raw_data[0:n_inputs, :]
rows = n_inputs

# Transform scalings values
scalings = raw_data[:, -1]  # last column
scalings = np.where(scalings > max_scaling, 0.0, (1.0 /
                    scalings))   # scaling between 0.0 and 1.0
# scalings = np.where(scalings > max_scaling, max_scaling, scalings)

# Create training and validation datasets

# q, dq, (x,y,z) of obstacle (firs, second, last columns excluded)
input = torch.Tensor(raw_data[:, 0:-1]).reshape(rows, cols-1)
target = torch.Tensor(scalings).reshape(rows, 1)

# random shuffle
indices = torch.randperm(input.size()[0])
input = input[indices]
target = target[indices]

train_size = int(rows)

train_dataset = torch.utils.data.TensorDataset(
    input[0:train_size, :], target[0:train_size])  # create your train datset

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

if load_net:
    NN = torch.load(nn_path)
else:
    NN = nn.Sequential(
        nn.Linear(dof+dof+3, 1000),
        nn.Tanh(),
        nn.Linear(1000, 1000),
        nn.Tanh(),
        nn.Linear(1000, 100),
        nn.Tanh(),
        nn.Linear(100, 100),
        nn.Tanh(),
        nn.Linear(100, 1),
        nn.Sigmoid()
    ).to(device)

print(NN)

# Define loss function and optimizer

if loss_fcn == "L1":
    criterion = torch.nn.L1Loss(reduction='sum')
else:
    criterion = torch.nn.MSELoss()

optimizer = optim.Adam(NN.parameters(), lr=lr, amsgrad=True)
# optimizer = optim.SGD(NN.parameters(), lr=lr, momentum=0.9)
# optimizer = optim.Adadelta(NN.parameters(), lr = lr)

# Train & Validation

# prepare for visualization
train_loss_over_epoches = []

plt.rcParams["figure.figsize"] = [12, 15]
ax = plt.GridSpec(2, 2)
ax.update(wspace=0.5, hspace=0.5)

ax0 = plt.subplot(ax[0, :])
ax1 = plt.subplot(ax[1, 0])
ax2 = plt.subplot(ax[1, 1])

for epoch in range(n_epochs):

    # Training phase
    # print("----- TRAINING -----")

    NN.train()  # set training mode

    train_loss_per_epoch = 0.0

    train_targets = []
    train_output = []
    train_predictions_errors = []

    for i, batch in enumerate(train_dataloader, 0):  # batches
        # get the inputs -> batch is a list of [inputs, targets]
        full_batch_input, batch_targets = batch
        batch_input = full_batch_input[:, 2:]

        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        predictions = NN(batch_input)

        # backpropagation
        loss = criterion(predictions, batch_targets)
        loss.backward()

        optimizer.step()

        # print current performance
        if loss_fcn == "L1":
            train_loss_per_epoch += loss.item()
        else:
            train_loss_per_epoch += (loss.item()*batch_targets.size()[0])

        train_output.extend(predictions.detach().cpu().numpy())
        train_targets.extend(batch_targets.detach().cpu().numpy())
        train_predictions_errors.extend(batch_targets.detach().cpu().numpy()-predictions.detach().cpu(
        ).numpy())

        max_error = max(abs(batch_targets.detach().cpu().numpy() -
                        predictions.detach().cpu().numpy()))
        print('[%d,] batch loss: %.6f max error: %.6f' %
              (epoch + 1, loss.item(), max_error))

    # Plot figures
    if True:

        if epoch % 500 == 499:
            train_loss_over_epoches = []

        train_loss_over_epoches.append(
            train_loss_per_epoch/len(train_dataloader))

        ax0.clear()
        ax0.set(xlabel="Epoches", ylabel="Loss",
                title="Training Loss")
        ax0.grid(True)
        ax0.plot(train_loss_over_epoches)

        ax1.clear()
        ax1.set(xlabel="Target", ylabel="Error",
                title="Prediction Error")
        ax1.grid(True)
        ax1.plot(train_targets, train_predictions_errors, '.', color='orange')

        ax2.clear()
        ax2.set(xlabel="Target", ylabel="Prediction",
                title="Target-Prediction")
        ax2.grid(True)
        ax2.plot(train_targets, train_output, '.', color='orange')

        plt.show(block=False)
        plt.draw()
        plt.pause(0.0001)
        plt.savefig(fig_path, dpi=300)

