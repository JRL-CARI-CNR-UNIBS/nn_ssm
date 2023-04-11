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

#include <ros/ros.h>
#include <memory>
#include <eigen3/Eigen/Eigen>
#include <Eigen/Dense>

namespace neural_network{

using data_type = double;
typedef Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic> MatrixXn;
typedef Eigen::Matrix<data_type, Eigen::Dynamic, 1> VectorXn;

class NeuralNetwork {
protected:
  const std::string layers_param_ = "/layer";
  const std::string inputs_param_ = "/inputs";
  const std::string bias_param_ = "/bias";
  const std::string nodes_param_ = "/nodes";
  const std::string weights_param_ = "/weights";
  const std::string activations_param_ = "/activation";

  std::vector<uint> topology_;
  std::vector<MatrixXn> weights_;
  std::vector<VectorXn> bias_;
  std::vector<std::function<data_type (const data_type&)>> activations_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  NeuralNetwork(){}

  inline void importFromParam(const ros::NodeHandle& nh, const std::string& name)
  {
    bias_.clear();
    weights_.clear();
    topology_.clear();
    activations_.clear();

    int n_inputs;
    if (nh.hasParam(name+inputs_param_))
    {
      if (not nh.getParam(name+inputs_param_,n_inputs))
        throw std::runtime_error("inputs param can't be read");
      else
        topology_.push_back(n_inputs);
    }
    else
      throw std::runtime_error("inputs param doesn't exist");

    uint i = 0;
    int nodes;
    std::string layer_name;
    std::string activation;
    VectorXn tmp_bias;
    std::vector<data_type> bias, weights;

    while(true && ros::ok())
    {
      layer_name = name+layers_param_+std::to_string(i);

      if(not nh.hasParam(layer_name))
        break;

      //layer's nodes
      if(not nh.getParam(layer_name+nodes_param_,nodes))
        throw std::runtime_error("nodes param can't be read");
      else
        topology_.push_back(nodes);

      //layer's bias
      if(not nh.getParam(layer_name+bias_param_,bias))
        throw std::runtime_error("bias param can't be read");
      else
      {
        tmp_bias = Eigen::Map<VectorXn, Eigen::Unaligned>(bias.data(), bias.size());
        bias_.push_back(tmp_bias);
      }

      //layer's activations
      if(not nh.getParam(layer_name+activations_param_,activation))
        throw std::runtime_error("activation param can't be read");
      else
      {
        if(activation == "tanh")
          activations_.push_back([](const data_type& x)->data_type{return std::tanh(x);});
        else if(activation == "sigmoid")
          activations_.push_back([](const data_type& x)->data_type{return (1.0/(1.0+exp(-x)));});
        else if(activation == "relu")
          activations_.push_back([](const data_type& x)->data_type{
            if(x<=0.0)
              return 0;
            else
              return x;
          });
        else
          throw std::runtime_error("activation not implemented yet");
      }

      //layer's weights
      if(not nh.getParam(layer_name+weights_param_,weights))
        throw std::runtime_error("weights param can't be read");
      else
      {
        uint n_weights_per_neuron = topology_[topology_.size()-2]; //number of nodes of the previous layer
        assert((weights.size()/n_weights_per_neuron) == topology_.back());

        Eigen::Map<MatrixXn> weights_matrix(weights.data(), topology_.back(), n_weights_per_neuron);
        weights_.push_back(weights_matrix);

        assert([&]() ->bool{
                 uint row = 0;
                 uint col = 0;
                 for(uint t=0;t<weights.size();t++)
                 {
                   if(weights[t] != weights_matrix(row,col))
                   return false;

                   col++;
                   if(col == n_weights_per_neuron)
                   {
                     col = 0;
                     row++;
                   }
                 }
                 return true;
               }());

//        for(uint j=0;j<topology_.back();j++)
//        {
//          std::vector<data_type>::const_iterator first = weights.begin()+j*n_weights_per_neuron;
//          std::vector<data_type>::const_iterator last  = first+n_weights_per_neuron;

//          tmp_weights = Eigen::Map<VectorXn, Eigen::Unaligned>(weights.data(), weights.size());

//          weights_matrix.conservativeResize(weights_matrix.rows()+1,Eigen::NoChange);
//          weights_matrix.row(weights_matrix.rows()-1) = tmp_weights;

//        }

//        tmp_weights = Eigen::Map<VectorXn, Eigen::Unaligned>(bias.data(), bias.size());
//        weights_.push_back(weights_matrix);
      }
    }
  }

  // function for forward propagation of data
  inline MatrixXn forward(MatrixXn inputs)
  {
    assert(activations_.size() == weights_.size());

    for(uint i = 0; i<weights_.size();i++)
    {
      assert(weights_[i].cols() == inputs  .rows());
      assert(weights_[i].rows() == bias_[i].rows());
      assert(bias_[i].cols() == 1);

      inputs = (weights_[i]*inputs).colwise() + bias_[i];
      inputs = inputs.unaryExpr(activations_[i]);
    }

    return inputs;
  }

  // storage objects for working of neural network
  /*
    use pointers when using std::vector<Class> as std::vector<Class> calls destructor of
    Class as soon as it is pushed back! when we use pointers it can't do that, besides
    it also makes our neural network class less heavy!! It would be nice if you can use
    smart pointers instead of usual ones like this
    */

};
}

