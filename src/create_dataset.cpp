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


#include <graph_core/informed_sampler.h>
#include <ssm15066_estimators/ssm15066_estimator2D.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <iostream>
#include <fstream>
#include <random>

struct Sample
{
  Eigen::VectorXd q;
  Eigen::VectorXd dq;
  std::vector<std::vector<double>> obstacles;

  double scaling;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "create_ssm_database");
  ros::NodeHandle nh;

  ros::AsyncSpinner spinner(4);
  spinner.start();

  // Get params
  std::string group_name;
  nh.getParam("group_name",group_name);

  std::string base_frame;
  nh.getParam("base_frame",base_frame);

  std::string tool_frame;
  nh.getParam("tool_frame",tool_frame);

  double max_step_size;
  nh.getParam("max_step_size",max_step_size);

  double max_cart_acc;
  nh.getParam("max_cart_acc",max_cart_acc);

  double t_r;
  nh.getParam("Tr",t_r);

  double min_distance;
  nh.getParam("min_distance",min_distance);

  double v_h;
  nh.getParam("v_h",v_h);

  int n_object;
  nh.getParam("n_object",n_object);

  int n_samples;
  nh.getParam("n_samples",n_samples);

  std::vector<std::string> poi_names;
  nh.getParam("poi_names",poi_names);

  std::vector<double> min_range;
  nh.getParam("min_range",min_range);

  std::vector<double> max_range;
  nh.getParam("max_range",max_range);

  // Create manipulator model, chain and ssm module
  moveit::planning_interface::MoveGroupInterface move_group(group_name);
  robot_model_loader::RobotModelLoaderPtr robot_model_loader = std::make_shared<robot_model_loader::RobotModelLoader>("robot_description");
  robot_model::RobotModelPtr robot_model = robot_model_loader->getModel();

  std::vector<std::string> joint_names = robot_model->getJointModelGroup(group_name)->getActiveJointModelNames();
  unsigned int dof = joint_names.size();
  Eigen::VectorXd lb(dof);
  Eigen::VectorXd ub(dof);

  for (unsigned int idx = 0; idx < dof; idx++)
  {
    const robot_model::VariableBounds& bounds = robot_model->getVariableBounds(joint_names.at(idx));
    if (bounds.position_bounded_)
    {
      lb(idx) = bounds.min_position_;
      ub(idx) = bounds.max_position_;
    }
  }

  pathplan::SamplerPtr sampler = std::make_shared<pathplan::InformedSampler>(lb, ub, lb, ub);

  Eigen::Vector3d grav; grav << 0, 0, -9.806;
  rosdyn::ChainPtr chain = rosdyn::createChain(*robot_model_loader->getURDF(),base_frame,tool_frame,grav);
  Eigen::VectorXd inv_max_speed = chain->getDQMax().cwiseInverse();

  ssm15066_estimator::SSM15066Estimator2DPtr ssm = std::make_shared<ssm15066_estimator::SSM15066Estimator2D>(chain,max_step_size);
  ssm->setMaxCartAcc(max_cart_acc,false);
  ssm->setReactionTime(t_r,false);
  ssm->setMinDistance(min_distance,false);
  ssm->setHumanVelocity(v_h,false);
  ssm->setPoiNames(poi_names);
  ssm->updateMembers();

  // Iterate over samples
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::cout<<"Database creation starts"<<std::endl;

  unsigned int iter;
  double r, distance, slowest_joint_time;
  Eigen::Vector3d obs_location;
  Eigen::VectorXd parent, child, connection, dq, delta_q;

  std::vector<Sample> samples;
  std::vector<double> obstacle;
  std::vector<std::vector<double>> obstacles;

  for(unsigned int i=0; i<n_samples; i++)
  {
    // Create obstacle locations
    obstacles.clear();
    ssm->clearObstaclesPositions();
    for(unsigned int j=0;j<n_object;j++)
    {
      r=dist(gen); obs_location[0] = min_range.at(0)+(max_range.at(0)-min_range.at(0))*r;
      r=dist(gen); obs_location[1] = min_range.at(1)+(max_range.at(1)-min_range.at(1))*r;
      r=dist(gen); obs_location[2] = min_range.at(2)+(max_range.at(2)-min_range.at(2))*r;

      ssm->addObstaclePosition(obs_location);

      obstacle.clear();
      obstacle = {obs_location[0],obs_location[1],obs_location[2]};
      obstacles.push_back(obstacle);
    }

    // Select a random connection and consider points along it
    parent = sampler->sample();
    child  = sampler->sample();

    connection= (parent-child);
    distance = connection.norm();

    if(distance>1.0)
      child = parent+(child-parent)*1.0/distance;

    iter = std::max(std::ceil((connection).norm()/max_step_size),1.0);
    delta_q = connection/iter;

    // Joints velocity vector
    slowest_joint_time = (inv_max_speed.cwiseProduct(connection)).cwiseAbs().maxCoeff();
    dq = connection/slowest_joint_time;

    for(unsigned int t=0;t<iter+1;t++)
    {
      Sample sample;
      sample.q = parent+t*delta_q;
      sample.dq = dq;
      sample.obstacles = obstacles;
      sample.scaling = ssm->computeScalingFactorAtQ(sample.q,dq);

      samples.push_back(sample);
    }
  }

  // Save params in a database
  std::ofstream file_params("ssm_database_creation_params.bin", std::ios::out | std::ios::binary);
  const size_t bufsize = 1024 * 1024;
  std::unique_ptr<char[]> buf_params;
  buf_params.reset(new char[bufsize]);

  file_params.rdbuf()->pubsetbuf(buf_params.get(), bufsize);

  file_params.write((char*) &group_name   , sizeof(group_name                  ));
  file_params.write((char*) &base_frame   , sizeof(base_frame                  ));
  file_params.write((char*) &tool_frame   , sizeof(tool_frame                  ));
  file_params.write((char*) &max_step_size, sizeof(max_step_size               ));
  file_params.write((char*) &max_cart_acc , sizeof(max_cart_acc                ));
  file_params.write((char*) &t_r          , sizeof(t_r                         ));
  file_params.write((char*) &min_distance , sizeof(min_distance                ));
  file_params.write((char*) &v_h          , sizeof(v_h                         ));
  file_params.write((char*) &n_object     , sizeof(n_object                    ));
  file_params.write((char*) &n_samples    , sizeof(n_samples                   ));
  file_params.write((char*) &poi_names[0] , sizeof(std::string)*poi_names.size());
  file_params.write((char*) &min_range[0] , sizeof(double)     *min_range.size());
  file_params.write((char*) &max_range[0] , sizeof(double)     *max_range.size());


  file_params.flush();
  file_params.close();

  // Save samples in a database
  std::ofstream file("ssm_database.bin", std::ios::out | std::ios::binary);
  std::unique_ptr<char[]> buf;
  buf.reset(new char[bufsize]);

  file.rdbuf()->pubsetbuf(buf.get(), bufsize);

  std::vector<double> tmp;
  std::vector<double> sample_vector;
  for(const Sample sample:samples)
  {
    sample_vector.clear();

    //q
    tmp.clear();
    tmp.resize(sample.q.size());
    Eigen::VectorXd::Map(&tmp[0], sample.q.size()) = sample.q;
    sample_vector.insert(sample_vector.end(),tmp.begin(),tmp.end());

    //dq
    tmp.clear();
    tmp.resize(sample.dq.size());
    Eigen::VectorXd::Map(&tmp[0], sample.dq.size()) = sample.dq;
    sample_vector.insert(sample_vector.end(),tmp.begin(),tmp.end());

    // scaling
    sample_vector.push_back(sample.scaling);

    //obstacles
    for(const std::vector<double>& obs:obstacles)
      sample_vector.insert(sample_vector.end(),obs.begin(),obs.end());

    file.write((char*)&sample_vector[0], sample_vector.size()*sizeof(double));
  }

  file.flush();
  file.close();

  return 0;
}
