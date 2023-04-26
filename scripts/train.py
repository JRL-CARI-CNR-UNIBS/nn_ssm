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

list_dataset_name = ["10k"]
list_n_epochs = [5000]
list_batch_size = [32]
lr_vector = [0.001]

list_dataset_name = ["10k","50k","100k","250k"]
list_n_epochs = [1000,3000,3000,10000]
list_batch_size = [32,32,64,128]
lr_vector = [0.001,0.001,0.001,0.001]

perc_train = 0.8
loss_fcn = ""

freq_batch = 10
freq_epoch = 10
freq_clear_plot = 100000

# Get paths
PATH = os.path.dirname(os.path.abspath(__file__)) + "/data/"
nn_path_shared = PATH + nn_name

for d in range(len(list_dataset_name)):
  dataset_name = "ssm_dataset_"+list_dataset_name[d]+".bin"
  n_epochs = list_n_epochs[d]
  batch_size = list_batch_size[d]
  lr = lr_vector[d]

  # Get paths
  dataset_path = PATH + dataset_name
  nn_path = PATH + list_dataset_name[d] + "_" + nn_name
  fig_path = PATH + list_dataset_name[d] + fig_name

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
  # Output: length, poi_first, poi_mid, poi_last, dq, v_safe first, v_safe mid, v_safe last, speed first, speed mid, speed last,
  # dist first, dist mid, dist last, scaling first, scaling mid, scaling last, scaling
  n_input = dof+dof+3
  n_output = 1+3*3+dof+13
  cols = n_input + n_output
  rows = int(length/cols)

  # from array to multi-dimensional-array
  raw_data = raw_data.reshape(rows, cols)

  input = torch.Tensor(raw_data[:, 0:n_input]).reshape(rows,n_input)
  target = torch.Tensor(raw_data[:, n_input:]).reshape(rows,n_output)

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
    nn.Linear(n_input, 1000),
    nn.Tanh(),
    nn.Linear(1000, n_output),
    nn.Sigmoid()
    ).to(device)
    load_net = True
     
    # NN = nn.Sequential(
    # nn.Linear(n_input, 1500),
    # nn.Tanh(),
    # nn.Dropout(0.5),
    # nn.Linear(1500, 1500),
    # nn.Tanh(),
    # nn.Dropout(0.3),
    # nn.Linear(1500, n_output),
    # nn.Sigmoid()
    # ).to(device)
    # load_net = True

    #         NN = nn.Sequential(
    #       nn.Linear(n_input, 1000),
    #       nn.Tanh(),
    #       nn.Dropout(0.1),
    #       nn.Linear(1000, 500),
    #       nn.Tanh(),
    #       nn.Dropout(0.01),
    #       nn.Linear(500, n_output),
    #       nn.Sigmoid()
    #   ).to(device)

    #       NN = nn.Sequential(
    #       nn.Linear(n_input, 4000),
    #       nn.Tanh(),
    #       nn.Dropout(0.05),
    #       nn.Linear(4000, 3000),
    #       nn.Tanh(),
    #       nn.Dropout(0.05),
    #       nn.Linear(3000, 2000),
    #       nn.Tanh(),
    #       nn.Dropout(0.05),
    #       nn.Linear(2000, 1000),
    #       nn.Tanh(),
    #       nn.Dropout(0.05),
    #       nn.Linear(1000, n_output),
    #       nn.Sigmoid()
    #   ).to(device)
    #   load_net = True

    #   NN = nn.Sequential(
    #   nn.Linear(n_input, 7000),
    #   nn.Tanh(),
    #   nn.Dropout(0.1),
    #   nn.Linear(7000, 3000),
    #   nn.Tanh(),
    #   nn.Dropout(0.05),
    #   nn.Linear(3000, 1000),
    #   nn.Tanh(),
    #   nn.Dropout(0.01),
    #   nn.Linear(1000, 100),
    #   nn.Tanh(),
    #   nn.Linear(100, n_output),
    #   nn.Sigmoid()
    # ).to(device)

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
  ax = plt.GridSpec(7, 2)
  ax.update(wspace=1, hspace=1.5)

  ax0 = plt.subplot(ax[0, :])
  ax1 = plt.subplot(ax[1, 0])
  ax2 = plt.subplot(ax[1, 1])
  ax3 = plt.subplot(ax[2, 0])
  ax4 = plt.subplot(ax[2, 1])

  ax5 = plt.subplot(ax[3, 0])
  ax6 = plt.subplot(ax[3, 1])
  ax7 = plt.subplot(ax[4, 0])
  ax8 = plt.subplot(ax[4, 1])

  ax9  = plt.subplot(ax[5, 0])
  ax10 = plt.subplot(ax[5, 1])
  ax11 = plt.subplot(ax[6, 0])
  ax12 = plt.subplot(ax[6, 1])

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
        #   loss = criterion(predictions, batch_targets)
          loss = criterion(predictions**3, batch_targets**3)
          loss.backward()

          optimizer.step()

          # print current performance
          train_loss_per_epoch += (loss.item()*batch_targets.size()[0])

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
              
              val_scaling_mid = []
              val_speed_mid = []
              val_dist_mid = []
              val_v_safe_mid = []
              val_length =  []
              val_poi_x =  []
              val_poi_y =  []
              val_poi_z =  []

              val_target_scaling_mid = []
              val_target_speed_mid = []
              val_target_dist_mid = []
              val_target_v_safe_mid = []

              val_target_length =  []
              val_target_poi_x =  []
              val_target_poi_y =  []
              val_target_poi_z =  []

              batches_loss = 0.0

              for i, batch in enumerate(val_dataloader, 0):
                  # get the inputs -> batch is a list of [inputs, targets]
                  batch_input, batch_targets = batch

                  batch_input = batch_input.to(device)
                  batch_targets = batch_targets.to(device)

                  predictions = NN(batch_input)
                #   loss = criterion(predictions, batch_targets)
                  loss = criterion(predictions**3, batch_targets**3)

                  val_loss_per_epoch += (loss.item()*batch_targets.size()[0])

                  batches_loss += loss.item()
                  if i % freq_batch == freq_batch-1:    # print every 1000 mini-batches
                      print('[%s -> %d, %5d] val batch loss: %.8f' %
                            (list_dataset_name[d], epoch + 1, i + 1, batches_loss/freq_batch))
                      batches_loss = 0.0

                  val_output.extend(predictions[:,-1].detach().cpu().numpy())
                  val_targets.extend(batch_targets[:,-1].detach().cpu().numpy())
                  val_predictions_errors.extend(batch_targets[:,-1].detach().cpu().numpy()-predictions[:,-1].detach().cpu().numpy())

                  val_scaling_mid.extend(predictions[:,-3].detach().cpu().numpy())
                  val_target_scaling_mid.extend(batch_targets[:,-3].detach().cpu().numpy())

                  val_dist_mid.extend(predictions[:,-6].detach().cpu().numpy())
                  val_target_dist_mid.extend(batch_targets[:,-6].detach().cpu().numpy())

                  val_speed_mid.extend(predictions[:,-9].detach().cpu().numpy())
                  val_target_speed_mid.extend(batch_targets[:,-9].detach().cpu().numpy())

                  val_v_safe_mid.extend(predictions[:,-12].detach().cpu().numpy())
                  val_target_v_safe_mid.extend(batch_targets[:,-12].detach().cpu().numpy())

                  val_length.extend(predictions[:,-29].detach().cpu().numpy())
                  val_target_length.extend(batch_targets[:,-29].detach().cpu().numpy())

                  val_poi_x.extend(predictions[:,-25].detach().cpu().numpy())
                  val_target_poi_x.extend(batch_targets[:,-25].detach().cpu().numpy())

                  val_poi_y.extend(predictions[:,-24].detach().cpu().numpy())
                  val_target_poi_y.extend(batch_targets[:,-24].detach().cpu().numpy())

                  val_poi_z.extend(predictions[:,-23].detach().cpu().numpy())
                  val_target_poi_z.extend(batch_targets[:,-23].detach().cpu().numpy())


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

          ax5.clear()
          ax5.set(xlabel="Target", ylabel="Prediction",
                  title="Speed mid")
          ax5.grid(True)
          ax5.plot(val_target_speed_mid, val_speed_mid, '.')

          ax6.clear()
          ax6.set(xlabel="Target", ylabel="Prediction",
                  title="Distance mid")
          ax6.grid(True)
          ax6.plot(val_target_dist_mid, val_dist_mid, '.')
          
          ax7.clear()
          ax7.set(xlabel="Target", ylabel="Prediction",
                  title="Scaling mid")
          ax7.grid(True)
          ax7.plot(val_target_scaling_mid, val_scaling_mid, '.')

          ax8.clear()
          ax8.set(xlabel="Target", ylabel="Prediction",
                  title="V_safe mid")
          ax8.grid(True)
          ax8.plot(val_target_v_safe_mid, val_v_safe_mid, '.')

          ax9.clear()
          ax9.set(xlabel="Target", ylabel="Prediction",
                  title="Length")
          ax9.grid(True)
          ax9.plot(val_target_length, val_length, '.')

          ax10.clear()
          ax10.set(xlabel="Target", ylabel="Prediction",
                  title="Poi X mid")
          ax10.grid(True)
          ax10.plot(val_target_poi_x, val_poi_x, '.')

          ax11.clear()
          ax11.set(xlabel="Target", ylabel="Prediction",
                  title="Poi Y mid")
          ax11.grid(True)
          ax11.plot(val_target_poi_y, val_poi_y, '.')

          ax12.clear()
          ax12.set(xlabel="Target", ylabel="Prediction",
                  title="Poi Z mid")
          ax12.grid(True)
          ax12.plot(val_target_poi_z, val_poi_z, '.')

          plt.show(block=False)
          plt.draw()
          plt.pause(0.0001)
          plt.savefig(fig_path, dpi=300)

      # Save NN at each epoch
      NN = NN.eval()
      torch.save(NN, nn_path_shared)
      torch.save(NN, nn_path)


