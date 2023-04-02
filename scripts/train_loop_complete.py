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
nn_name = "nn_ssm_complete.pt"
list_dataset_name = ["1k","10k","25","50k","75k","100k","125","150k","200","250k","500k"]
list_n_epochs = [2000,2000,2000,3000,3000,3000,3000,3000,3000,3000,3000]
list_batch_size = [64,64,64,128,128,264,264,264,328,328,328]

perc_train = 0.8
loss_fcn = ""
lr = 0.001

freq_batch = 10
freq_epoch = 10
freq_clear_plot = 200

# Get paths
PATH = os.path.dirname(os.path.abspath(__file__)) + "/data/"
nn_path_shared = PATH + nn_name

for d in range(len(list_dataset_name)):
  dataset_name = "ssm_dataset_"+list_dataset_name[d]+".bin"
  n_epochs = list_n_epochs[d]
  batch_size = list_batch_size[d]

  # Get paths
  dataset_path = PATH + dataset_name
  nn_path = PATH + list_dataset_name[d] + "_" + nn_name
  fig_path = PATH + list_dataset_name[d] + fig_name

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
  print("speed tensor")
  speed_tensor = torch.Tensor(raw_data[:,-3]).reshape(rows, 1)
  print(speed_tensor[0:10,:])
  print("distance tensor")
  distance_tensor = torch.Tensor(raw_data[:,-2]).reshape(rows, 1)
  print(distance_tensor[0:10,:])

  target = torch.cat((speed_tensor,distance_tensor,scalings_tensor), -1)

  # random shuffle
  indices = torch.randperm(input.size()[0])
  input = input[indices]
  target = target[indices]

  train_size = int(rows*perc_train)
  val_size = rows-train_size

  train_dataset = torch.utils.data.TensorDataset(
      input[0:train_size, :], target[0:train_size,:])  # create your train datset
  val_dataset = torch.utils.data.TensorDataset(
      input[train_size:, :], target[train_size:,:])   # create your validation datset

  train_dataloader = torch.utils.data.DataLoader(
      dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
  val_dataloader = torch.utils.data.DataLoader(
      dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

  if load_net:
      NN = torch.load(nn_path_shared)
  else:
      NN = nn.Sequential(
          nn.Linear(dof+dof+3, 1000),
          nn.Tanh(),
          # nn.Dropout(p=0.1),
          nn.Linear(1000, 1000),
          nn.Tanh(),
          # nn.Dropout(p=0.1),
          nn.Linear(1000, 100),
          nn.Tanh(),
          # nn.Dropout(p=0.1),
          nn.Linear(100, 3),
          nn.Sigmoid()
      ).to(device)
      load_net = True

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
  ax = plt.GridSpec(6, 2)
  ax.update(wspace=0.5, hspace=0.5)

  ax0 = plt.subplot(ax[0, :])
  ax1 = plt.subplot(ax[1, 0])
  ax2 = plt.subplot(ax[1, 1])
  ax3 = plt.subplot(ax[2, 0])
  ax4 = plt.subplot(ax[2, 1])
  ax5 = plt.subplot(ax[3, 0])
  ax6 = plt.subplot(ax[3, 1])
  ax7 = plt.subplot(ax[4, 0])
  ax8 = plt.subplot(ax[4, 1])

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
      train_speed = []
      train_speed_error = []
      train_distance = []
      train_distance_error = []
      train_target_speed = []
      train_target_distance = []

      for i, batch in enumerate(train_dataloader, 0):  # batches
          # get the inputs -> batch is a list of [inputs, targets]
          batch_input, batch_targets = batch

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
              print('[%s -> %d, %5d] train batch loss: %.8f' %
                    (list_dataset_name[d], epoch + 1, i + 1, batches_loss/freq_batch))
              batches_loss = 0.0

          if epoch % freq_epoch == freq_epoch-1:
            train_output.extend(predictions[:,2].detach().cpu().numpy())
            train_targets.extend(batch_targets[:,2].detach().cpu().numpy())
            train_predictions_errors.extend(batch_targets[:,2].detach().cpu().numpy()-predictions[:,2].detach().cpu().numpy())
            train_speed.extend(predictions[:,0].detach().cpu().numpy())
            train_target_speed.extend(batch_targets[:,0].detach().cpu().numpy())
            train_speed_error.extend(batch_targets[:,0].detach().cpu().numpy()-predictions[:,0].detach().cpu().numpy())
            train_distance.extend(predictions[:,1].detach().cpu().numpy())
            train_target_distance.extend(batch_targets[:,1].detach().cpu().numpy())
            train_distance_error.extend(batch_targets[:,1].detach().cpu().numpy()-predictions[:,1].detach().cpu().numpy())

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
                  batch_input, batch_targets = batch

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
                      print('[%s -> %d, %5d] val batch loss: %.8f' %
                            (list_dataset_name[d], epoch + 1, i + 1, batches_loss/freq_batch))
                      batches_loss = 0.0

                  val_output.extend(predictions[:,2].detach().cpu().numpy())
                  val_targets.extend(batch_targets[:,2].detach().cpu().numpy())
                  val_predictions_errors.extend(batch_targets[:,2].detach().cpu().numpy()-predictions[:,2].detach().cpu(
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

          ax5.clear()
          ax5.set(xlabel="Speed", ylabel="Error",
                  title="Speed Error")
          ax5.grid(True)
          ax5.plot(train_target_speed, train_speed_error, '.', color='orange')

          ax6.clear()
          ax6.set(xlabel="Speed", ylabel="Predicted speed",
                  title="Target-Prediction")
          ax6.grid(True)
          ax6.plot(train_target_speed, train_speed, '.', color='orange')

          ax7.clear()
          ax7.set(xlabel="Distance", ylabel="Error",
                  title="Distance Error")
          ax7.grid(True)
          ax7.plot(train_target_distance, train_distance_error, '.', color='orange')

          ax8.clear()
          ax8.set(xlabel="Distance", ylabel="Predicted distance",
                  title="Target-Prediction")
          ax8.grid(True)
          ax8.plot(train_target_distance, train_distance, '.', color='orange')

          plt.show(block=False)
          plt.draw()
          plt.pause(0.0001)
          plt.savefig(fig_path, dpi=300)

      # Save NN at each epoch
      NN = NN.eval()
      torch.save(NN, nn_path_shared)
      torch.save(NN, nn_path)
