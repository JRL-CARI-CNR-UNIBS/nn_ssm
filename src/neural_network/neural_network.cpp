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

#include <nn_ssm/neural_network/neural_network.h>

namespace neural_network
{
// constructor of neural network class
NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate)
{
  this->topology = topology;
  this->learningRate = learningRate;
  for (uint i = 0; i < topology.size(); i++) {
    // initialize neuron layers
    if (i == topology.size() - 1)
      neuronLayers.push_back(new RowVector(topology[i]));
    else
      neuronLayers.push_back(new RowVector(topology[i] + 1));

    // initialize cache and delta vectors
    cacheLayers.push_back(new RowVector(neuronLayers.size()));
    deltas.push_back(new RowVector(neuronLayers.size()));

    // vector.back() gives the handle to recently added element
    // coeffRef gives the reference of value at that place
    // (using this as we are using pointers here)
    if (i != topology.size() - 1) {
      neuronLayers.back()->coeffRef(topology[i]) = 1.0;
      cacheLayers.back()->coeffRef(topology[i]) = 1.0;
    }

    // initialize weights matrix
    if (i > 0) {
      if (i != topology.size() - 1) {
        weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
        weights.back()->setRandom();
        weights.back()->col(topology[i]).setZero();
        weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
      }
      else {
        weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
        weights.back()->setRandom();
      }
    }
  }
};

}
