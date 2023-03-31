import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Params
dof = 6
load_net = True
max_scaling = 1000
fig_name = str(dof)+"dof.png"
nn_name = "nn_ssm.pt"
dataset_name = "ssm_dataset_10k.bin"
n_epochs = 5000
perc_train = 0.8
loss_fcn = ""
lr = 0.001

freq_batch = 10
freq_epoch = 10
freq_clear_plot = 200

# Batch size
if dataset_name == "ssm_dataset_1k.bin":
    batch_size = 64
elif dataset_name == "ssm_dataset_10k.bin":
    batch_size = 128
elif dataset_name == "ssm_dataset_50k.bin":
    batch_size = 264
elif dataset_name == "ssm_dataset_75k.bin":
    batch_size = 264
elif dataset_name == "ssm_dataset_100k.bin":
    batch_size = 328
elif dataset_name == "ssm_dataset_150k.bin":
    batch_size = 328
else:
    batch_size = 656


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

# for t in range(rows):
#     if raw_data[t,0] <= 0.0 and raw_data[t,-1] != 1.0:
#         print(f"speed {raw_data[t,0]:.3f}, target {raw_data[t,-1]:.3f}")

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

train_size = int(rows*perc_train)
val_size = rows-train_size

train_dataset = torch.utils.data.TensorDataset(
    input[0:train_size, :], target[0:train_size])  # create your train datset
val_dataset = torch.utils.data.TensorDataset(
    input[train_size:, :], target[train_size:])   # create your validation datset

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

if load_net:
    NN = torch.load(nn_path)
else:
    NN = nn.Sequential(
        nn.Linear(dof+dof+3, 1000),
        nn.Tanh(),
        # nn.Dropout(p=0.1),
        nn.Linear(1000, 1000),
        nn.Tanh(),
        # nn.Dropout(p=0.1),
        nn.Linear(1000, 1000),
        nn.Tanh(),
        # nn.Dropout(p=0.1),
        nn.Linear(1000, 100),
        nn.Tanh(),
        # nn.Dropout(p=0.1),
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
# optimizer = optim.SGD(NN.parameters(), lr=lr, momentum=0.7)

# Train & Validation

# prepare for visualization
val_loss_over_epoches = []
train_loss_over_epoches = []

plt.rcParams["figure.figsize"] = [12, 15]
ax = plt.GridSpec(4, 2)
ax.update(wspace=0.5, hspace=0.5)

ax0 = plt.subplot(ax[0, :])
ax1 = plt.subplot(ax[1, 0])
ax2 = plt.subplot(ax[1, 1])
ax3 = plt.subplot(ax[2, 0])
ax4 = plt.subplot(ax[2, 1])

for epoch in range(n_epochs):

    # Training phase
    print("----- TRAINING -----")

    NN = NN.train()  # set training mode

    train_loss_per_epoch = 0.0
    val_loss_per_epoch = 0.0
    batches_loss = 0.0

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

        batches_loss += loss.item()
        if i % freq_batch == freq_batch-1:    # print every freq_batch mini-batches
            print('[%d, %5d] train batch loss: %.8f' %
                  (epoch + 1, i + 1, batches_loss/freq_batch))
            batches_loss = 0.0

        if epoch % 10 == 9:
            train_output.extend(predictions.detach().cpu().numpy())
            train_targets.extend(batch_targets.detach().cpu().numpy())
            train_predictions_errors.extend(batch_targets.detach().cpu().numpy()-predictions.detach().cpu(
            ).numpy())

        # for j in range(len(predictions)):
        #     if (epoch > 500 and abs(predictions[j] - batch_targets[j]) >0.1):
        #         pr = predictions[j].detach().cpu().numpy()
        #         tr = batch_targets[j].detach().cpu().numpy()
        #         sp = full_batch_input[j,0].detach().cpu().numpy()
        #         di = full_batch_input[j,1].detach().cpu().numpy()
        #         print(
        #             f"vel {sp}, distance {di}, predicted {pr}, target {tr}")

    if epoch % freq_epoch == freq_epoch-1:  # Validation every freq_epoch epochs
        # Validation phase
        print("----- VALIDATION -----")

        with torch.no_grad():  # evaluation does not need gradients computation
            NN = NN.eval()  # set evaluation mode

            val_targets = []
            val_output = []
            val_predictions_errors = []
            batches_loss = 0.0

            for i, batch in enumerate(val_dataloader, 0):
                # get the inputs -> batch is a list of [inputs, targets]
                full_batch_input, batch_targets = batch
                batch_input = full_batch_input[:, 2:]

                batch_input = batch_input.to(device)
                batch_targets = batch_targets.to(device)

                predictions = NN(batch_input)
                loss = criterion(predictions, batch_targets)

                if loss_fcn == "L1":
                    val_loss_per_epoch += loss.item()
                else:
                    val_loss_per_epoch += (loss.item()*batch_targets.size()[0])

                batches_loss += loss.item()
                if i % freq_batch == freq_batch-1:    # print every 1000 mini-batches
                    print('[%d, %5d] val batch loss: %.8f' %
                          (epoch + 1, i + 1, batches_loss/freq_batch))
                    batches_loss = 0.0

                val_output.extend(predictions.detach().cpu().numpy())
                val_targets.extend(batch_targets.detach().cpu().numpy())
                val_predictions_errors.extend(batch_targets.detach().cpu().numpy()-predictions.detach().cpu(
                ).numpy())
    print("-----------------------------------------------------------")
    # print(f"max err {max(train_predictions_errors)}")

    if epoch % freq_clear_plot == freq_clear_plot-1: # reset
        train_loss_over_epoches = []
        val_loss_over_epoches = []

    # Plot figures
    if epoch % freq_epoch == freq_epoch-1:
        train_loss_over_epoches.append(
            train_loss_per_epoch/len(train_dataloader))
        val_loss_over_epoches.append(val_loss_per_epoch/len(val_dataloader))

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
        ax1.plot(val_targets, val_predictions_errors, '.')

        ax2.clear()
        ax2.set(xlabel="Target", ylabel="Prediction",
                title="Target-Prediction")
        ax2.grid(True)
        ax2.plot(val_targets, val_output, '.')

        ax3.clear()
        ax3.set(xlabel="Target", ylabel="Error",
                title="Prediction Error")
        ax3.grid(True)
        ax3.plot(train_targets, train_predictions_errors, '.', color='orange')

        ax4.clear()
        ax4.set(xlabel="Target", ylabel="Prediction",
                title="Target-Prediction")
        ax4.grid(True)
        ax4.plot(train_targets, train_output, '.', color='orange')

        plt.show(block=False)
        plt.draw()
        plt.pause(0.0001)
        plt.savefig(fig_path, dpi=300)

    # Save NN at each epoch
    NN = NN.eval()
    torch.save(NN, nn_path)
