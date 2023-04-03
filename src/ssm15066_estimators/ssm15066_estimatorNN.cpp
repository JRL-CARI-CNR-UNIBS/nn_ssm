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

SSM15066EstimatorNN::SSM15066EstimatorNN(const rosdyn::ChainPtr &chain, const double& max_step_size):
  SSM15066Estimator2D(chain,max_step_size)
{
  setDevice();
}

SSM15066EstimatorNN::SSM15066EstimatorNN(const rosdyn::ChainPtr &chain, const double &max_step_size, const Eigen::Matrix<double,3,Eigen::Dynamic> &obstacles_positions):
  SSM15066Estimator2D(chain,max_step_size,obstacles_positions)
{
  setDevice();
}

SSM15066EstimatorNN::SSM15066EstimatorNN(const rosdyn::ChainPtr &chain, const std::string path, const double& max_step_size):
  SSM15066Estimator2D(chain,max_step_size)
{
  setDevice();
  loadModel(path);
}

SSM15066EstimatorNN::SSM15066EstimatorNN(const rosdyn::ChainPtr &chain, const std::string path, const double &max_step_size, const Eigen::Matrix<double,3,Eigen::Dynamic> &obstacles_positions):
  SSM15066Estimator2D(chain,max_step_size,obstacles_positions)
{
  setDevice();
  loadModel(path);
}

std::vector<double> SSM15066EstimatorNN::feedforward(const std::vector<double>& input, const unsigned int& n_samples)
{
  unsigned int cols = input.size()/n_samples;
  at::Tensor tensor =  torch::from_blob(input.data(),{n_samples,cols}, opt_);
  assert(tensor.device() == device_);

  std::vector<torch::jit::IValue> tensor_in;
  tensor_in.push_back(tensor);

  at::Tensor tensor_out = model_.forward(tensor_in).toTensor();

  std::vector<double> output(tensor_out.data_ptr<double>(), tensor_out.data_ptr<double>() + tensor_out.numel());
  return output;
}

double SSM15066EstimatorNN::computeScalingFactor(const Eigen::VectorXd& q1, const Eigen::VectorXd& q2)
{
  if(obstacles_positions_.cols()==0)  //no obstacles in the scene
    return 1.0;

  if(verbose_>0)
  {
    ROS_ERROR_STREAM("number of obstacles: "<<obstacles_positions_.cols()<<", number of poi: "<<poi_names_.size());
    for(unsigned int i=0;i<obstacles_positions_.cols();i++)
      ROS_ERROR_STREAM("obs location -> "<<obstacles_positions_.col(i).transpose());
  }

  /* Compute the time of each joint to move from q1 to q2 at its maximum speed and consider the longest time */
  Eigen::VectorXd connection_vector = (q2-q1);
  double slowest_joint_time = (inv_max_speed_.cwiseProduct(connection_vector)).cwiseAbs().maxCoeff();

  /* The "slowest" joint will move at its highest speed while the other ones will
   * move at (t_i/slowest_joint_time)*max_speed_i, where slowest_joint_time >= t_i */
  Eigen::VectorXd dq = connection_vector/slowest_joint_time;

  if(verbose_>0)
    ROS_ERROR_STREAM("joint velocity "<<dq.norm());

  unsigned int iter = std::max(std::ceil((connection_vector).norm()/max_step_size_),1.0);

  Eigen::VectorXd q;
  Eigen::VectorXd delta_q = connection_vector/iter;

  double max_scaling_factor_of_q;
  double sum_scaling_factor = 0.0;

  for(unsigned int i=0;i<iter+1;i++)
  {
    //CREA VETTORE CON SAMPLES
    q = q1+i*delta_q;
    max_scaling_factor_of_q = computeScalingFactorAtQ(q,dq); //CHIAMA RETE NEURALE
    //ESTRAI RISULTATI

    if(max_scaling_factor_of_q == std::numeric_limits<double>::infinity())
      return std::numeric_limits<double>::infinity();
    else
      sum_scaling_factor += max_scaling_factor_of_q;
  }

  // return the average scaling factor
  double res = sum_scaling_factor/((double) iter+1);
  return res;
}

double SSM15066EstimatorNN::computeScalingFactorAtQ(const Eigen::VectorXd& q, const Eigen::VectorXd& dq,  double& tangential_speed, double& distance)
{
  // TODO
}

pathplan::CostPenaltyPtr SSM15066EstimatorNN::clone()
{
  // TODO
  SSM15066EstimatorNNPtr ssm_cloned = std::make_shared<SSM15066EstimatorNN>(chain_->clone(),max_step_size_,obstacles_positions_);

  pathplan::CostPenaltyPtr clone = ssm_cloned;

  return clone;
}


}
