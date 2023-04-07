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

#include <vector>
#include <iostream>
#include <eigen3/Eigen/Eigen>

typedef double Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

class NeuralNetwork {
protected:
  std::vector<RowVector*> neuronLayers; // stores the different layers of out network
  std::vector<RowVector*> cacheLayers; // stores the unactivated (activation fn not yet applied) values of layers
  std::vector<RowVector*> deltas; // stores the error contribution of each neurons
  std::vector<Matrix*> weights; // the connection weights itself
  Scalar learningRate;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));

  // function for forward propagation of data
  void propagateForward(RowVector& input);

  // function for backward propagation of errors made by neurons
  void propagateBackward(RowVector& output);

  // function to calculate errors made by neurons in each layer
  void calcErrors(RowVector& output);

  // function to update the weights of connections
  void updateWeights();

  // function to train the neural network give an array of data points
  void train(std::vector<RowVector*> data);

  // storage objects for working of neural network
  /*
    use pointers when using std::vector<Class> as std::vector<Class> calls destructor of
    Class as soon as it is pushed back! when we use pointers it can't do that, besides
    it also makes our neural network class less heavy!! It would be nice if you can use
    smart pointers instead of usual ones like this
    */

};
