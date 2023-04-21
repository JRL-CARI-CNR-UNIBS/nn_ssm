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

#include <nn_ssm/ssm15066_estimators/ssm15066_estimatorNN.h>

namespace ssm15066_estimator
{
SSM15066EstimatorNN::SSM15066EstimatorNN(const rosdyn::ChainPtr &chain, const neural_network::NeuralNetworkPtr& nn,
                                         const Eigen::Vector3d& obs_min_range, const Eigen::Vector3d& obs_max_range):
  SSM15066Estimator2D(chain),nn_(nn),obs_min_range_(obs_min_range),obs_max_range_(obs_max_range)
{
  init();
}

SSM15066EstimatorNN::SSM15066EstimatorNN(const rosdyn::ChainPtr &chain, const neural_network::NeuralNetworkPtr& nn,
                                         const Eigen::Matrix<double,3,Eigen::Dynamic>& obstacles_positions,
                                         const Eigen::Vector3d& obs_min_range, const Eigen::Vector3d& obs_max_range):
  SSM15066Estimator2D(chain),nn_(nn),obs_min_range_(obs_min_range),obs_max_range_(obs_max_range)
{
  init();
  setObstaclesPositions(obstacles_positions);
}

void SSM15066EstimatorNN::init()
{
  lb_ = chain_->getQMin();
  inv_q_limits_ = (chain_->getQMax()-lb_).cwiseInverse();
  obs_inv_range_ = (obs_max_range_-obs_min_range_).cwiseInverse();
}

void SSM15066EstimatorNN::setObstaclesPositions(const Eigen::Matrix<double,3,Eigen::Dynamic>& obstacles_positions)
{
  SSM15066Estimator::setObstaclesPositions(obstacles_positions);

  for(uint i_obs = 0; i_obs<obstacles_positions_.cols(); i_obs++)
    obstacles_positions_.col(i_obs) = (obstacles_positions_.col(i_obs)-obs_min_range_).cwiseProduct(obs_inv_range_);
}
void SSM15066EstimatorNN::addObstaclePosition(const Eigen::Vector3d& obstacle_position)
{
  Eigen::Vector3d obstacle_position_normalized = (obstacle_position-obs_min_range_).cwiseProduct(obs_inv_range_);
  SSM15066Estimator::addObstaclePosition(obstacle_position_normalized);
}

double SSM15066EstimatorNN::computeScalingFactor(const Eigen::VectorXd& q1, const Eigen::VectorXd& q2)
{
  if(obstacles_positions_.cols()==0)  //no obstacles in the scene
    return 1.0;

  Eigen::VectorXd q1_scaled = (inv_q_limits_).cwiseProduct(q1-lb_);
  Eigen::VectorXd q2_scaled = (inv_q_limits_).cwiseProduct(q2-lb_);

  neural_network::MatrixXn input (q1.size()*2+3,obstacles_positions_.cols());  //rows -> q1+q2+(x,y,z)       cols -> n_obstacles
  neural_network::MatrixXn output(13+q1.size()  ,obstacles_positions_.cols()); //rows -> scaling + dq + ...  cols -> n_obstacles
  for(Eigen::Index i_obs=0;i_obs<obstacles_positions_.cols();i_obs++)
    input.col(i_obs) << q1_scaled,q2_scaled,obstacles_positions_.col(i_obs)[0],obstacles_positions_.col(i_obs)[1],obstacles_positions_.col(i_obs)[2];

  output = nn_->forward(input);

  // First element of each column is the scaling expressed between 0 and 1. Find the worst case and reverse it!
  double scaling = output.row(0).minCoeff();
  if(scaling<1e-03)
    scaling = std::numeric_limits<double>::infinity();
  else
    scaling = 1.0/scaling;

  return scaling;
}

pathplan::CostPenaltyPtr SSM15066EstimatorNN::clone()
{
  SSM15066EstimatorNNPtr ssm_cloned = std::make_shared<SSM15066EstimatorNN>(chain_->clone(),nn_->clone(),obs_min_range_,obs_max_range_);

  ssm_cloned->setPoiNames(poi_names_);
  ssm_cloned->setMaxStepSize(max_step_size_);
  ssm_cloned->setObstaclesPositions(obstacles_positions_);

  ssm_cloned->setMaxCartAcc(max_cart_acc_,false);
  ssm_cloned->setMinDistance(min_distance_,false);
  ssm_cloned->setReactionTime(reaction_time_,false);
  ssm_cloned->setHumanVelocity(human_velocity_,false);

  ssm_cloned->updateMembers();

  pathplan::CostPenaltyPtr clone = ssm_cloned;

  return clone;
}

}
