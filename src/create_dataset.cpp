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
#include <ros/package.h>

struct Sample
{
  Eigen::VectorXd q;
  Eigen::VectorXd dq;
  double speed;
  double distance;
  std::vector<std::vector<double>> obstacles;

  double scaling;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "create_ssm_database");
  ros::NodeHandle nh;

  ros::AsyncSpinner spinner(4);
  spinner.start();

  std::srand(std::time(NULL));

  // Get params
  bool only_one_conf_per_conn;
  nh.getParam("only_one_conf_per_conn",only_one_conf_per_conn);

  bool scale_input;
  nh.getParam("scale_input",scale_input);

  int n_objects;
  nh.getParam("n_objects",n_objects);

  int n_iter;
  nh.getParam("n_iter",n_iter);

  int n_test_per_obs;
  nh.getParam("n_test_per_obs",n_test_per_obs);

  std::string group_name;
  nh.getParam("group_name",group_name);

  std::string base_frame;
  nh.getParam("base_frame",base_frame);

  std::string tool_frame;
  nh.getParam("tool_frame",tool_frame);

  std::string database_name;
  nh.getParam("database_name",database_name);

  int n_divisions;
  nh.getParam("n_divisions",n_divisions);

  double max_cart_acc;
  nh.getParam("max_cart_acc",max_cart_acc);

  double t_r;
  nh.getParam("Tr",t_r);

  double min_distance;
  nh.getParam("min_distance",min_distance);

  double v_h;
  nh.getParam("v_h",v_h);

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
  Eigen::VectorXd max_speed  = chain->getDQMax();
  Eigen::VectorXd min_speed  = -max_speed;
  Eigen::VectorXd inv_max_speed  = max_speed.cwiseInverse();
  Eigen::VectorXd inv_q_limits  = (ub-lb).cwiseInverse();
  Eigen::VectorXd inv_speed_limits  = (max_speed-min_speed).cwiseInverse();

  ssm15066_estimator::SSM15066Estimator2DPtr ssm = std::make_shared<ssm15066_estimator::SSM15066Estimator2D>(chain,0.001);
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
  ros::Duration(5).sleep();

  double x, y, z, slowest_joint_time;
  Eigen::Vector3d obs_location;
  Eigen::VectorXd parent, child, connection, dq, delta_q, q, q_scaled, dq_scaled;

  std::vector<Sample> samples;
  std::vector<double> obs;
  std::vector<std::vector<double>> obstacles;

  bool progress_bar_full = false;
  unsigned int progress = 0;

  for(unsigned int i=0; i<n_iter; i++)
  {
    // Create obstacle locations
    obstacles.clear();
    ssm->clearObstaclesPositions();

    for(unsigned int j=0;j<n_objects;j++)
    {
      x = dist(gen); obs_location[0] = min_range.at(0)+(max_range.at(0)-min_range.at(0))*x;
      y = dist(gen); obs_location[1] = min_range.at(1)+(max_range.at(1)-min_range.at(1))*y;
      z = dist(gen); obs_location[2] = min_range.at(2)+(max_range.at(2)-min_range.at(2))*z;

      ssm->addObstaclePosition(obs_location);

      if(scale_input)
        obs = {x,y,z};
      else
        obs = {obs_location[0],obs_location[1],obs_location[2]};

      obstacles.push_back(obs);
    }

    for(unsigned int k=0; k<n_test_per_obs; k++) //Test multiple robot configurations and velocities with the same set of obstacles
    {
      // Select a random connection and consider points along it
      parent = sampler->sample();
      child  = sampler->sample();

      connection = (child-parent);
      delta_q = connection/((double)n_divisions);

      assert((delta_q*((double)n_divisions)-connection).norm()<1e-06);

      // Joints velocity vector
      slowest_joint_time = (inv_max_speed.cwiseProduct(connection)).cwiseAbs().maxCoeff();
      dq = connection/slowest_joint_time;

      for(unsigned int t=0;t<n_divisions+1;t++) //Test multiple robot configurations along the same connection
      {
        q = parent+t*delta_q;

        // Scale input
        if(scale_input)
        {
          q_scaled = (inv_q_limits).cwiseProduct(q-lb);
          dq_scaled = (inv_speed_limits).cwiseProduct(dq-min_speed);
        }
        else
        {
          q_scaled = q;
          dq_scaled = dq;
        }

        Sample sample;
        sample.q = q_scaled;
        sample.dq = dq_scaled;
        sample.obstacles = obstacles;
        sample.scaling  = ssm->computeScalingFactorAtQ(q,dq,sample.speed,sample.distance); //q and dq, not q_scaled and dq_scaled!

        assert([&]() ->bool{
                 if(sample.speed<=0 && sample.scaling!=1.0)
                 {
                   ROS_RED_STREAM("speed not ok! -> distance "<<sample.distance<<" speed "<< sample.speed<< " scaling "<<sample.scaling);
                   return false;
                 }
                 if(sample.distance<=min_distance && sample.speed>0.0 && sample.scaling!=std::numeric_limits<double>::infinity())
                 {
                   ROS_RED_STREAM("distance not ok! -> distance "<<sample.distance<<" speed "<< sample.speed<< " scaling "<<sample.scaling);
                   return false;
                 }
                 return true;
               }());

        samples.push_back(sample);

        assert(sample.scaling>=1.0);
        assert([&]() ->bool{
                 if(scale_input)
                 {
                   for(unsigned int l=0;l<q_scaled.size();l++)
                   {
                     if(q_scaled[l]>1.0 || q_scaled[l]<0.0)
                     {
                       return false;
                     }
                   }

                   for(unsigned int l=0;l<dq_scaled.size();l++)
                   {
                     if(dq_scaled[l]>1.0 || dq_scaled[l]<0.0)
                     {
                       return false;
                     }
                   }

                   for(const std::vector<double>& tmp_vector:obstacles)
                   {
                     for(const double& tmp:tmp_vector)
                     {
                       if(tmp>1.0 || tmp<0.0)
                       {
                         return false;
                       }
                     }
                   }
                 }
                 return true;
               }());

        if(only_one_conf_per_conn)
          break;
      }

      assert([&]() ->bool{
               if(not only_one_conf_per_conn)
               {
                 if((q-child).norm()>1e-04)
                 {
                   ROS_INFO_STREAM("q "<<q.transpose());
                   ROS_INFO_STREAM("child "<<child.transpose());
                   ROS_INFO_STREAM("err "<<(q-child).norm());
                   return false;
                 }
               }
               return true;
             }());
    }

    progress = std::ceil(((double)(i+1.0))/((double)n_iter)*100.0);
    if(progress%1 == 0 && not progress_bar_full)
    {
      std::string output = "\r[";

      for(unsigned int j=0;j<progress/5.0;j++)
        output = output+"=";

      output = output+">] ";
      output = "\033[1;41;42m"+output+std::to_string(progress)+"%\033[0m";  //1->bold, 37->foreground white, 42->background green

      if(progress == 100)
      {
        progress_bar_full = true;
        output = output+"\033[1;5;32m Succesfully completed!\033[0m\n";
      }

      std::cout<<output;
    }
  }

  // Save params in a database
  std::string path = ros::package::getPath("nn_ssm");
  path = path+"/scripts/data/";

  std::ofstream file_params;
  file_params.open(path+database_name+"_creation_params.bin", std::ios::out | std::ios::binary);

  const size_t bufsize = 1024 * 1024;
  std::unique_ptr<char[]> buf_params;
  buf_params.reset(new char[bufsize]);

  file_params.rdbuf()->pubsetbuf(buf_params.get(), bufsize); n_test_per_obs;

  file_params.write((char*) &scale_input   , sizeof(scale_input                 ));
  file_params.write((char*) &group_name    , sizeof(group_name                  ));
  file_params.write((char*) &base_frame    , sizeof(base_frame                  ));
  file_params.write((char*) &tool_frame    , sizeof(tool_frame                  ));
  file_params.write((char*) &n_divisions   , sizeof(n_divisions                 ));
  file_params.write((char*) &max_cart_acc  , sizeof(max_cart_acc                ));
  file_params.write((char*) &t_r           , sizeof(t_r                         ));
  file_params.write((char*) &min_distance  , sizeof(min_distance                ));
  file_params.write((char*) &v_h           , sizeof(v_h                         ));
  file_params.write((char*) &n_objects     , sizeof(n_objects                   ));
  file_params.write((char*) &n_iter        , sizeof(n_iter                      ));
  file_params.write((char*) &n_test_per_obs, sizeof(n_test_per_obs              ));
  file_params.write((char*) &poi_names[0]  , sizeof(std::string)*poi_names.size());
  file_params.write((char*) &min_range[0]  , sizeof(double)     *min_range.size());
  file_params.write((char*) &max_range[0]  , sizeof(double)     *max_range.size());

  file_params.flush();
  file_params.close();

  // Save samples in a database
  std::ofstream file;
  file.open(path+database_name+".bin", std::ios::out | std::ios::binary);
  std::unique_ptr<char[]> buf;
  buf.reset(new char[bufsize]);

  file.rdbuf()->pubsetbuf(buf.get(), bufsize);

  std::vector<double> tmp;
  std::vector<double> sample_vector;
  for(const Sample& sample:samples)
  {
    sample_vector.clear();

    //speed
    sample_vector.push_back(sample.speed);

    //distance
    sample_vector.push_back(sample.distance);

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

    //obstacles
    for(const std::vector<double>& obs:sample.obstacles)
      sample_vector.insert(sample_vector.end(),obs.begin(),obs.end());

    // scaling
    sample_vector.push_back(sample.scaling);

    //    std::string txt = "obs: ";
    //    for(const std::vector<double>& obs:sample.obstacles)
    //      txt = txt+"("+std::to_string(obs[0])+","+std::to_string(obs[1])+","+std::to_string(obs[2])+") ";

    //    ROS_INFO_STREAM("q: "<<sample.q.transpose()<<" | dq: "<<sample.dq.transpose()<<" | "<<txt+"| scaling: "<<sample.scaling);

    file.write((char*)&sample_vector[0], sample_vector.size()*sizeof(double));
  }

  file.flush();
  file.close();

  ROS_INFO_STREAM("Total number of samples -> "<<samples.size());

  return 0;
}
