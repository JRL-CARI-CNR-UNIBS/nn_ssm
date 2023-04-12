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

class NeuralNetwork;
typedef std::shared_ptr<NeuralNetwork> NeuralNetworkPtr;

/**
 * @brief The NeuralNetwork class implements a C++ ROS-compatible class for fully-connected feedforward neural networks. This class is able to read parameters of a pre-trained
 * neural networks from ROS Param or files (TODO). You don't need to specify the network topology because it is able to automatically derive it.
 * Input and output are matrices in which each column is a sample/prediction.
 */

class NeuralNetwork: public std::enable_shared_from_this<NeuralNetwork>
{
protected:
  /**
   * @brief These consts are the params names used to save the neural networks parameters into the ROS Param Server
   */
  const std::string layers_param_ = "/layer";
  const std::string inputs_param_ = "/inputs";
  const std::string bias_param_ = "/bias";
  const std::string neurons_param_ = "/neurons";
  const std::string weights_param_ = "/weights";
  const std::string activations_param_ = "/activation";

  /**
   * @brief topology_ is the structure of the neural network. It saves the number of inputs as first element and the number of neurons for each layer then:
   *  [n_inputs, neurons_hidden1, neurons_hidden2, ... , neurons_output_layer]
   */
  std::vector<uint> topology_;
  /**
   * @brief weights_ is a vector of matrices of weights. The i-th matrix in the vector is the weights matrix of the i-th layer. Each j-th row of the i-th matrix
   * is the vector of weights of the j-th neuron of the i-th layer.
   */
  std::vector<MatrixXn> weights_;

  /**
   * @brief bias_ is a vector of bias vectors. The i-th element is the bias vector of the i-th layer.
   */
  std::vector<VectorXn> bias_;

  /**
   * @brief activations_ is a vector of activation functions. The i-th element is the activation applied to all the neurons of the i-th layer.
   */
  std::vector<std::function<data_type (const data_type&)>> activations_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief NeuralNetwork constructor. Then you need to use importFromParam to build the neural network.
   */
  NeuralNetwork(){}

  /**
   * @brief NeuralNetwork costructor
   * @param topology topology vector,in the form of [n_inputs, neurons_hidden1, neurons_hidden2, ... , neurons_output_layer]
   * @param weights vector of matrix containing the weights of each layer. The i-th matrix in the vector is the weights matrix
   * of the i-th layer. Each j-th row of the i-th matrix is the vector of weights of the j-th neuron of the i-th layer.
   * @param biasis a vector of bias vectors. The i-th element is the bias vector of the i-th layer.
   * @param activationsis a vector of activation functions. The i-th element is the activation applied to all the neurons of the i-th layer.
   */
  NeuralNetwork(const std::vector<uint>& topology, const std::vector<MatrixXn>& weights,
                const std::vector<VectorXn>& bias, const std::vector<std::function<data_type (const data_type&)>>& activations):
    topology_(topology),weights_(weights),bias_(bias),activations_(activations)
  {}

  /**
   * @brief importFromParam read the parameters of the neural network from ROS Param.
   * @param nh is a ROS node handle
   * @param name is the namespace of the parameters. Es: my_nn is the namespace -> /my_nn/layer0/weights, /my_nn/layer1/weights ..
   */
  inline void importFromParam(const ros::NodeHandle& nh, const std::string& name = "")
  {
    bias_.clear();
    weights_.clear();
    topology_.clear();
    activations_.clear();

    int n_inputs;
    if(nh.hasParam(name+inputs_param_))
    {
      if (not nh.getParam(name+inputs_param_,n_inputs))
        throw std::runtime_error("inputs param can't be read");
      else
        topology_.push_back(n_inputs);
    }
    else
      throw std::runtime_error("inputs param doesn't exist");

    uint layer_number = 0;
    int neurons;
    VectorXn tmp_bias;
    std::string layer_name;
    std::string activation;
    uint n_weights_per_neuron;
    std::vector<data_type> bias, weights;

    while(true && ros::ok())
    {
      layer_name = name+layers_param_+std::to_string(layer_number);

      if(not nh.hasParam(layer_name))
      {
        if(layer_number==0) //neural network not defined yet
          throw std::runtime_error("neural network not properly defined");
        else    //at least a layer has been defined
          break;
      }

      //layer's neurons
      if(not nh.getParam(layer_name+neurons_param_,neurons))
        throw std::runtime_error("neurons param can't be read");
      else
        topology_.push_back(neurons);

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
        n_weights_per_neuron = topology_[topology_.size()-2]; //number of neurons of the previous layer
        assert((weights.size()/n_weights_per_neuron) == topology_.back());

        Eigen::Map<Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> weights_matrix(weights.data(), topology_.back(), n_weights_per_neuron);
        weights_.push_back(weights_matrix);

        assert([&]() ->bool{
                 uint row = 0;
                 uint col = 0;
                 for(uint t=0;t<weights.size();t++)
                 {
                   if(weights[t] != weights_matrix(row,col))
                   {
                     ROS_ERROR("weights ");
                     for(const data_type& w:weights)
                     ROS_ERROR_STREAM("w "<<w);

                     ROS_WARN("matrix ");
                     for(uint r=0;r<weights_matrix.rows();r++)
                     {
                       ROS_WARN_STREAM(weights_matrix.row(r));
                     }
                     return false;
                   }

                   col++;
                   if(col == n_weights_per_neuron)
                   {
                     col = 0;
                     row++;
                   }
                 }
                 return true;
               }());
        layer_number++;
      }
    }
  }

  /**
   * @brief forward is the forward function of the neural network, to be used to evaluate the inputs
   * @param inputs is a matrix of inputs, in which each column is a sample
   * @return the matrix of output, in which each column is a prediction
   */
  inline MatrixXn forward(MatrixXn inputs)
  {
    if(weights_.empty())
      throw std::runtime_error("Neural Network not properly initialized");

    assert(activations_.size() == weights_.size());

    // number of layers == weights_.size() (we have a matrix of weights for each layer)
    for(uint layer = 0; layer<weights_.size();layer++)
    {
      assert(weights_[layer].cols() == inputs  .rows());
      assert(weights_[layer].rows() == bias_[layer].rows());
      assert(bias_[layer].cols() == 1);

      // | weight11 weight12 weight13 .. |*|sample11 sample21 .. |    weights -> each row is associated to a neuron
      // | weight21 weight22 weight23 .. | |sample12 sample22 .. |    samples -> each col is a sample
      //                                   |sample13 sample23 .. |

      inputs = (weights_[layer]*inputs).colwise() + bias_[layer];  //bias is applied column-wise
      inputs = inputs.unaryExpr(activations_[layer]);              // activation is applied element-wise
    }

    return inputs;
  }

  inline NeuralNetworkPtr clone()
  {
    return std::make_shared<neural_network::NeuralNetwork>(topology_,weights_,bias_,activations_);
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

