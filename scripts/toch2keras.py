import os
import numpy as np
import torch
import torch.nn as nn
from pytorch2keras import pytorch_to_keras
from torch.autograd import Variable
import onnxruntime as rt
import onnx
from onnx2keras import onnx_to_keras
import tensorflow as tf

device = "cpu"
nn_name = "nn_torch2keras.pt"

n_inputs = 2
n_outputs = 2

model = nn.Sequential(
    nn.Linear(n_inputs, 2),
    nn.Sigmoid(),
    nn.Linear(2, n_outputs),
    nn.Sigmoid(),
).to(device)

model.eval()

PATH = os.path.dirname(os.path.abspath(__file__)) + "/data/"
nn_path_shared = PATH + nn_name
torch.save(model, nn_path_shared)

model = torch.load(nn_path_shared)

# Create pt2keras object
from pt2keras import Pt2Keras
converter = Pt2Keras()

# convert model
# model can be both pytorch nn.Module or 
# string path to .onnx file. E.g. 'model.onnx'
input_shape = (n_inputs)
keras_model: tf.keras.Model = converter.convert(model, input_shape)

x = torch.randn(n_inputs, requires_grad=True)

onnx_path = PATH+"model.onnx"
model_onnx = torch.onnx.export(model, x, onnx_path, verbose=False, input_names = ['input'], output_names = ['output'])

# sess = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
# input_name = sess.get_inputs()[0].name

# # Note: The input must be of the same shape as the shape of x during # the model export part. i.e. second argument in this function call: torch.onnx.export()
# onnxPredictions = sess.run(None, {input_name: x.detach().numpy()})[0]

# Load ONNX model
onnx_model = onnx.load(onnx_path)
# Call the converter (input will be equal to the input_names parameter that you defined during exporting)
k_model = onnx_to_keras(onnx_model, ['input'], name_policy='renumerate', verbose=False)
k_model.summary()

# print(f'pytorch {model(x)}')
# print(f'onnx {onnxPredictions}')
# print(f'keras {k_model.predict(x).detach().numpy()}')
