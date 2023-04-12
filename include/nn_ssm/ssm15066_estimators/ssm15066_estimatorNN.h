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

#include <ssm15066_estimators/ssm15066_estimator2D.h>
#include <nn_ssm/neural_networks/neural_network.h>

namespace ssm15066_estimator
{
class SSM15066EstimatorNN;
typedef std::shared_ptr<SSM15066EstimatorNN> SSM15066EstimatorNNPtr;

/**
  * @brief The SSM15066EstimatorNN class uses a Neural Network (NN) to approximate with a faster computation SSM15066Estimator2D.
  * It uses a neural network trained to estimate the lambda given q_parent, q_child and (x,y,z) of an obstacle. As output, it gives the inverse of lambda.
  * Note that these class members are set but never used because they are intrinsic in the trained neural network:
  *   - chain_
  *   - poi_names_
  *   - links_names_
  *   - inv_max_speed_
  *   - human_velocity_
  *   - reaction_time_
  *   - max_cart_acc_
  *   - min_distance_
  *   - max_step_size_
  *   - term1_
  *   - term2_
  * If you want to change them you should train another neural network
  */
class SSM15066EstimatorNN: public SSM15066Estimator2D
{
protected:
  neural_network::NeuralNetworkPtr nn_;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SSM15066EstimatorNN(const rosdyn::ChainPtr &chain, const neural_network::NeuralNetworkPtr& nn);
  SSM15066EstimatorNN(const rosdyn::ChainPtr &chain, const neural_network::NeuralNetworkPtr& nn,
                      const Eigen::Matrix<double,3,Eigen::Dynamic>& obstacles_positions);


  virtual double computeScalingFactor(const Eigen::VectorXd& q1, const Eigen::VectorXd& q2) override;
  virtual pathplan::CostPenaltyPtr clone() override;
};

}
