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

#include <nn_ssm/neural_network/feedforward_nn.h>

namespace neural_network
{
FeedForwardNN::FeedForwardNN(const std::vector<unsigned int>& nodes, const std::vector<Activation>& activations)
{
  createModel(nodes,activations);
}

FeedForwardNN::FeedForwardNN(const std::vector<unsigned int>& nodes, const std::vector<Activation>& activations, const std::string& path)
{
  createModel(nodes,activations);
  loadModel(path);
}

void FeedForwardNN::createModel(const std::vector<unsigned int>& nodes, const std::vector<Activation>& activations)
{
  torch::nn::Sequential model;
  for(size_t i=0;i<nodes.size()-1;i++)
  {
    model->push_back(torch::nn::Linear(nodes[i],nodes[i+1]));

    switch(activations[i])
    {
    case Activation::ReLU:
      model->push_back(torch::nn::ReLU());
      break;
    case Activation::TanH:
      model->push_back(torch::nn::Tanh());
      break;
    case Activation::Sigmoid:
      model->push_back(torch::nn::Sigmoid());
      break;
    default:
      throw std::runtime_error("Activation %s not implemented yet", activations[i]);
    }
  }

  model_ = model;
}

}
