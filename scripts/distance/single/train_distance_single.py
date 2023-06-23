import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Params
dof = 6
load_net = False
fig_name = str(dof)+"dof_distance_single.png"
nn_name = "nn_ssm_distance_single.pt"

full_dataset = False

list_dataset_name = ["1000k"]
list_n_epochs = [5000]
list_batch_size = [256]
lr_vector = [0.001]

perc_train = 0.8
loss_fcn = ""

freq_batch = 10
freq_epoch = 10
freq_clear_plot = 100000

# Get paths
PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
nn_path_shared = PATH + nn_name

for d in range(len(list_dataset_name)):
  dataset_name = "ssm_dataset_distance_single_"+list_dataset_name[d]+".bin"
  n_epochs = list_n_epochs[d]
  batch_size = list_batch_size[d]
  lr = lr_vector[d]

  # Get paths
  dataset_path = PATH + dataset_name
  nn_path = PATH + list_dataset_name[d] + "_" + nn_name
  fig_path_distance = PATH + list_dataset_name[d] + fig_name
  fig_path_distance2 = PATH + list_dataset_name[d] + "_fk_"+ fig_name

  # Get cpu or gpu device for training
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using {device} device")

  if device == "cuda":
      if torch.cuda.device_count()>1:
        device = torch.device('cuda:1')
      print(f"=> {torch.cuda.get_device_name(device)}")

  # Load dataset
  raw_data = np.fromfile(dataset_path, dtype='float')
  length = raw_data.size

  # Input: parent, child, (x,y,z) of obstacle,
  # Output: distance
  n_input = dof+3
  n_output = 1+3
  cols = n_input + n_output
  rows = int(length/cols)

  # from array to multi-dimensional-array
  raw_data = raw_data.reshape(rows, cols)

  input = torch.Tensor(raw_data[:, 0:n_input]).reshape(rows,n_input)
  target = torch.Tensor(raw_data[:, n_input:]).reshape(rows,n_output)

# random shuffle
#   indices = torch.randperm(input.size()[0])
#   input = input[indices]
#   target = target[indices]

  train_size = int(rows*perc_train)
  val_size = rows-train_size

  train_dataset = torch.utils.data.TensorDataset(
      input[0:train_size, :], target[0:train_size,:])  # create your train datset
  val_dataset = torch.utils.data.TensorDataset(
      input[train_size:, :], target[train_size:,:])   # create your validation datset

  if full_dataset:
     batch_size_train = train_size
     batch_size_val = val_size
  else:
     batch_size_train = batch_size
     batch_size_val = batch_size

  train_dataloader = torch.utils.data.DataLoader(
      dataset=train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=True)
  val_dataloader = torch.utils.data.DataLoader(
      dataset=val_dataset, batch_size=batch_size_val, shuffle=True, drop_last=True)

  if load_net:
      NN = torch.load(nn_path_shared)
  else:
    NN = nn.Sequential(
    nn.Linear(n_input, 100),
    nn.Tanh(),
    nn.Linear(100, 100),
    nn.Tanh(),
    nn.Linear(100, 100),
    nn.Tanh(),
    nn.Linear(100, 100),
    nn.Tanh(),
    nn.Linear(100, n_output),
    nn.Sigmoid()
    ).to(device)
    load_net = True

  print(NN)

  # Define loss function and optimizer

  if loss_fcn == "L1":
      criterion = torch.nn.L1Loss()
  else:
      criterion = torch.nn.MSELoss()

  optimizer = optim.Adam(NN.parameters(), lr=lr)
#   optimizer = optim.Adam(NN.parameters(), lr=lr, amsgrad=True)
  # optimizer = optim.SGD(NN.parameters(), lr=lr, momentum=0.7)

  # Train & Validation

  # prepare for visualization
  val_loss_over_epoches = []
  train_loss_over_epoches = []

  plt.rcParams["figure.figsize"] = [12, 15]

  plt.figure(1)
  ax = plt.GridSpec(3, 2)
  ax.update(wspace=0.5, hspace=0.5)

  ax0 = plt.subplot(ax[0, :])
  ax1 = plt.subplot(ax[1, 0])
  ax2 = plt.subplot(ax[1, 1])
  ax3 = plt.subplot(ax[2, 0])
  ax4 = plt.subplot(ax[2, 1])

  plt.figure(2)
  ax = plt.GridSpec(3, 2)
  ax.update(wspace=0.5, hspace=0.5)

  ax5  = plt.subplot(ax[0, 0])
  ax6  = plt.subplot(ax[0, 1])
  ax7  = plt.subplot(ax[1, 0])
  ax8  = plt.subplot(ax[1, 1])
  ax9  = plt.subplot(ax[2, 0])
  ax10 = plt.subplot(ax[2, 1])

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
          batch_input, batch_targets = batch

          batch_input = batch_input.to(device)
          batch_targets = batch_targets.to(device)

          # zero the parameter gradients
          optimizer.zero_grad()
          
          # forward
          predictions = NN(batch_input)

          # backpropagation
          loss = criterion(predictions, batch_targets)
        #   loss = criterion(predictions**3, batch_targets**3)
          loss.backward()

          optimizer.step()

          # print current performance
          train_loss_per_epoch += loss.item()

          batches_loss += loss.item()
          if i % freq_batch == freq_batch-1:    # print every freq_batch mini-batches
              print('[%s -> %d, %5d] train batch loss: %.8f' %
                    (list_dataset_name[d], epoch + 1, i + 1, batches_loss/freq_batch))
              batches_loss = 0.0

          if epoch % freq_epoch == freq_epoch-1:
            train_output.extend(predictions[:,-1].detach().cpu().numpy())
            train_targets.extend(batch_targets[:,-1].detach().cpu().numpy())
            train_predictions_errors.extend(batch_targets[:,-1].detach().cpu().numpy()-predictions[:,-1].detach().cpu().numpy())

      if epoch % freq_epoch == freq_epoch-1:  # Validation every freq_epoch epochs
          # Validation phase
          print("----- VALIDATION -----")

          with torch.no_grad():  # evaluation does not need gradients computation
              NN = NN.eval()  # set evaluation mode

              val_targets = []
              val_output = []
              val_predictions_errors = []

              val_x_targets= []
              val_x_err = []
              val_x_output = []

              val_y_targets = []
              val_y_err = []
              val_y_output = []

              val_z_targets = []
              val_z_err = []
              val_z_output = []

              batches_loss = 0.0

              for i, batch in enumerate(val_dataloader, 0):
                  # get the inputs -> batch is a list of [inputs, targets]
                  batch_input, batch_targets = batch

                  batch_input = batch_input.to(device)
                  batch_targets = batch_targets.to(device)

                  predictions = NN(batch_input)
                  loss = criterion(predictions, batch_targets)
                #   loss = criterion(predictions**3, batch_targets**3)
                  
                  val_loss_per_epoch += loss.item()

                  batches_loss += loss.item()
                  if i % freq_batch == freq_batch-1:    # print every 1000 mini-batches
                      print('[%s -> %d, %5d] val batch loss: %.8f' %
                            (list_dataset_name[d], epoch + 1, i + 1, batches_loss/freq_batch))
                      batches_loss = 0.0

                  val_output.extend(predictions[:,-1].detach().cpu().numpy())
                  val_targets.extend(batch_targets[:,-1].detach().cpu().numpy())
                  val_predictions_errors.extend(batch_targets[:,-1].detach().cpu().numpy()-predictions[:,-1].detach().cpu().numpy())

                  val_x_output.extend(predictions[:,-4].detach().cpu().numpy())
                  val_x_targets.extend(batch_targets[:,-4].detach().cpu().numpy())
                  val_x_err.extend(batch_targets[:,-4].detach().cpu().numpy()-predictions[:,-4].detach().cpu().numpy())

                  val_y_output.extend(predictions[:,-3].detach().cpu().numpy())
                  val_y_targets.extend(batch_targets[:,-3].detach().cpu().numpy())
                  val_y_err.extend(batch_targets[:,-3].detach().cpu().numpy()-predictions[:,-3].detach().cpu().numpy())

                  val_z_output.extend(predictions[:,-2].detach().cpu().numpy())
                  val_z_targets.extend(batch_targets[:,-2].detach().cpu().numpy())
                  val_z_err.extend(batch_targets[:,-2].detach().cpu().numpy()-predictions[:,-2].detach().cpu().numpy())

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
          txt = "Epoches (x "+str(freq_epoch)+")"
          ax0.set(xlabel=txt, ylabel="Loss",
                  title="Training ("+str(train_loss_over_epoches[-1])+") and Validation Loss ("+str(val_loss_over_epoches[-1])+")")
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

          plt.figure(1)
          plt.show(block=False)
          plt.draw()
          plt.pause(0.0001)
          plt.savefig(fig_path_distance, dpi=300)

          ax5.clear()
          ax5.set(xlabel="Target", ylabel="Prediction",
                  title="X Error")
          ax5.grid(True)
          ax5.plot(val_x_targets, val_x_err, '.')

          ax6.clear()
          ax6.set(xlabel="Target", ylabel="Prediction",
                  title="X")
          ax6.grid(True)
          ax6.plot(val_x_targets, val_x_output, '.')

          ax7.clear()
          ax7.set(xlabel="Target", ylabel="Prediction",
                  title="Y Error")
          ax7.grid(True)
          ax7.plot(val_y_targets, val_y_err, '.')

          ax8.clear()
          ax8.set(xlabel="Target", ylabel="Prediction",
                  title="Y")
          ax8.grid(True)
          ax8.plot(val_y_targets, val_y_output, '.')

          ax9.clear()
          ax9.set(xlabel="Target", ylabel="Prediction",
                  title="Z Error")
          ax9.grid(True)
          ax9.plot(val_z_targets, val_z_err, '.')

          ax10.clear()
          ax10.set(xlabel="Target", ylabel="Prediction",
                  title="Z")
          ax10.grid(True)
          ax10.plot(val_z_targets, val_z_output, '.')

          plt.figure(2)
          plt.show(block=False)
          plt.draw()
          plt.pause(0.0001)
          plt.savefig(fig_path_distance2, dpi=300)

      # Save NN at each epoch
      NN = NN.eval()
      torch.save(NN, nn_path_shared)
      torch.save(NN, nn_path)
      if epoch % 5000 == 4999:
        torch.save(NN, nn_path+"_"+str(epoch/1000)+"k_epochs")


