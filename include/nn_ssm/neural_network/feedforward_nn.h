#pragma once
/*
Copyright (c) 2023, Cesare Tonola University of Brescia c.tonola001@unibs.it
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <torch/torch.h>
#include <torch/script.h>
#include <nn_ssm/neural_network/util.h>

namespace neural_network
{
class FeedForwardNN;
typedef std::shared_ptr<FeedForwardNN> FeedForwardNNPtr;

/**
 * @brief The FeedForwardNN class represents a feedforward neural network.
 * Basically it builds a neural network given number of layers, nodes for layers and activation functions.
 * Then, it loads the parameters from a pre-trained neural network.
 * This class is supposed to be used only for inference. It is based on LibTorch.
 */
class FeedForwardNN: public FeedForwardNN
{
protected:

  /**
   * @brief model_ is the sequential model
   */
  torch::nn::Sequential model_;

  /**
   * @brief path_ is the absolute path to the pre-trained neural network
   */
  std::string path_;

  /**
   * @brief cuda_available_ is true when the model can run on cuda
   */
  bool cuda_available_;

  /**
   * @brief device_ is the device used to do inferences
   */
  torch::Device device_;

  void setDevice()
  {
    cuda_available_ = torch::cuda::is_available();
    torch::Device device(cuda_available_ ? torch::kCUDA : torch::kCPU);
    device_ = device;
  }

  /**
   * @brief createModel creates the feedforward model given the number of nodes of each layer and the activations.
   * @param nodes is the vector containing the number of nodes for each layer (including inputs and ouputs)
   * @param activations is a vector containing the activations for each layer. Note that activations size = nodes size -1
   */
  void createModel(const std::vector<unsigned int>& nodes, const std::vector<Activation>& activations);

public:
  FeedForwardNN(const std::vector<unsigned int>& nodes, const std::vector<Activation>& activations);
  FeedForwardNN(const std::vector<unsigned int>& nodes, const std::vector<Activation>& activations, const std::string& path);

  /**
   * @brief loadModel loads the pre-trained neural network model
   * @param path is the absolute path to the pretrained model
   */
  void loadModel(const std::string& path)
  {
    path_ = path;
    torch::load(model_,path);

    model_->eval();
    model_->to(device_);
  }
};

}
