import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import onnx
import onnx_tf
from onnx_tf.backend import prepare
from model_utils import save_model
import numpy as np
import tensorflow as tf
import keras

device = "cpu"
nn_name = "nn_torch2tensor.pt"

n_inputs = 1
n_outputs = 1

model = nn.Sequential(
    nn.Linear(n_inputs, 2),
    nn.Sigmoid(),
    nn.Linear(2, n_outputs),
    nn.Sigmoid(),
).to(device)

model.eval()

PATH = os.path.dirname(os.path.abspath(__file__)) + "/data/"
nn_path_shared = PATH + nn_name
onnx_path = PATH+"model.onnx"
tf_path = PATH+"model.pb"

torch.save(model, nn_path_shared)

model = torch.load(nn_path_shared)
model.eval()

input = torch.Tensor([1])
print(f'model output {model(input)}')

x_data =  torch.randn([n_inputs])
sample = np.array(x_data) 
sample_tensor=torch.from_numpy(sample.reshape(1,-1)).float()

torch.onnx.export(model, sample_tensor, onnx_path, export_params=True, input_names = ['input'], output_names = ['output'])

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_path)

# save_model(model_tmp, PATH+'model_weights.json')
