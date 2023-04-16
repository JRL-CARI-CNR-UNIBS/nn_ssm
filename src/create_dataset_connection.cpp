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
#include <graph_core/parallel_moveit_collision_checker.h>
#include <moveit_msgs/GetPlanningScene.h>
#include <iostream>
#include <fstream>
#include <random>
#include <ros/package.h>

struct Sample
{
  //Input
  Eigen::VectorXd parent, child;
  std::vector<double> obstacle;

  //Output
  Eigen::VectorXd dq;
  double scaling, scaling_first, scaling_mid, scaling_last, speed_first, speed_mid, speed_last, distance_first, distance_mid, distance_last,
  v_safe_first, v_safe_mid, v_safe_last;

  friend std::ostream& operator<<(std::ostream& os, const Sample& sample)
  {
    std::stringstream input, output;

    input<< "\n Input: \n parent -> "<<sample.parent.transpose()<<" child -> "<<sample.child.transpose()
         <<"\n (x,y,z) obstacle "<<sample.obstacle[0]<<", "<<sample.obstacle[1]<<", "<<sample.obstacle[2];

    output<< "\n Output:\n scaling: "<<sample.scaling<<" scaling first "<<sample.scaling_first<<" scaling mid "<<sample.scaling_mid<<" scaling last "<<sample.scaling_last;
    output<<"\n speed first "<<sample.speed_first<<" speed mid "<<sample.speed_mid<<" speed last "<<sample.speed_last;
    output<<"\n dist first "<<sample.distance_first<<" dist mid "<<sample.distance_mid<<" dist last "<<sample.distance_last;
    output<<"\n v_safe first "<<sample.v_safe_first<<" v_safe mid "<<sample.v_safe_mid<<" v_safe last "<<sample.v_safe_last;

    os<<input.str()<<output.str();
    return os;
  }
};

void random_replace(std::vector<Sample>& v, const Sample& sample)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> p(0.0, 1.0);
  if(p(gen)>=0.5)
  {
    std::uniform_int_distribution<size_t> dis(0, std::distance(v.begin(), v.end()) - 1);
    size_t idx = dis(gen);
    assert(idx<v.size() && idx>=0);
    v[idx] = sample;
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "create_ssm_database_connection");
  ros::NodeHandle nh;

  ros::AsyncSpinner spinner(4);
  spinner.start();

  std::srand(std::time(NULL));

  // Get params
  int n_objects;
  nh.getParam("n_objects",n_objects);

  int n_iter;
  nh.getParam("n_iter",n_iter);

  std::string group_name;
  nh.getParam("group_name",group_name);

  std::string base_frame;
  nh.getParam("base_frame",base_frame);

  std::string tool_frame;
  nh.getParam("tool_frame",tool_frame);

  std::string database_name;
  nh.getParam("database_name",database_name);

  double max_cart_acc;
  nh.getParam("max_cart_acc",max_cart_acc);

  double t_r;
  nh.getParam("Tr",t_r);

  double min_safe_distance;
  nh.getParam("min_distance",min_safe_distance);

  double v_h;
  nh.getParam("v_h",v_h);

  std::vector<std::string> poi_names;
  nh.getParam("poi_names",poi_names);

  std::vector<double> min_range;
  nh.getParam("min_range",min_range);

  std::vector<double> max_range;
  nh.getParam("max_range",max_range);

  ros::ServiceClient ps_client=nh.serviceClient<moveit_msgs::GetPlanningScene>("/get_planning_scene");

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

  planning_scene::PlanningScenePtr planning_scene = std::make_shared<planning_scene::PlanningScene>(robot_model);
  moveit_msgs::GetPlanningScene ps_srv;
  if (!ps_client.waitForExistence(ros::Duration(10)))
  {
    ROS_ERROR("unable to connect to /get_planning_scene");
    return 1;
  }

  if (!ps_client.call(ps_srv))
  {
    ROS_ERROR("call to srv not ok");
    return 1;
  }

  if (!planning_scene->setPlanningSceneMsg(ps_srv.response.scene))
  {
    ROS_ERROR("unable to update planning scene");
    return 1;
  }

  pathplan::CollisionCheckerPtr checker = std::make_shared<pathplan::ParallelMoveitCollisionChecker>(planning_scene, group_name, 5, 0.005);

  pathplan::SamplerPtr sampler = std::make_shared<pathplan::InformedSampler>(lb, ub, lb, ub);

  Eigen::Vector3d grav; grav << 0, 0, -9.806;
  rosdyn::ChainPtr chain = rosdyn::createChain(*robot_model_loader->getURDF(),base_frame,tool_frame,grav);
  Eigen::VectorXd max_speed = chain->getDQMax();
  Eigen::VectorXd min_speed = -max_speed;
  Eigen::VectorXd inv_max_speed = max_speed.cwiseInverse();
  Eigen::VectorXd inv_q_limits = (ub-lb).cwiseInverse();
  Eigen::VectorXd inv_speed_limits = (max_speed-min_speed).cwiseInverse();

  ssm15066_estimator::SSM15066Estimator2DPtr ssm = std::make_shared<ssm15066_estimator::SSM15066Estimator2D>(chain,0.001);
  ssm->setMaxCartAcc(max_cart_acc,false);
  ssm->setReactionTime(t_r,false);
  ssm->setMinDistance(min_safe_distance,false);
  ssm->setHumanVelocity(v_h,false);
  ssm->setPoiNames(poi_names);
  ssm->updateMembers();

  ssm->setDatasetCreation(true);

  // Iterate over samples
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::cout<<"Database creation starts"<<std::endl;
  ros::Duration(5).sleep();

  double x, y, z, slowest_joint_time;
  Eigen::Vector3d obs_location;
  Eigen::VectorXd parent, child, connection, dq, parent_scaled, child_scaled, dq_scaled;

  std::vector<Sample> samples, v_scaling_1, v_scaling_0, v_scaling_0_01, v_scaling_01_02, v_scaling_02_03, v_scaling_03_04,
      v_scaling_04_05, v_scaling_05_06, v_scaling_06_07, v_scaling_07_08, v_scaling_08_09, v_scaling_09_1;

  std::vector<double> obs;

  std::vector<int> vectors_fill(12,0);
  size_t n_samples_per_vector = (size_t)std::ceil(n_iter)/12;

  while(true && ros::ok())
  {
    // Create obstacle locations
    ssm->clearObstaclesPositions();

    x = dist(gen); obs_location[0] = min_range.at(0)+(max_range.at(0)-min_range.at(0))*x;
    y = dist(gen); obs_location[1] = min_range.at(1)+(max_range.at(1)-min_range.at(1))*y;
    z = dist(gen); obs_location[2] = min_range.at(2)+(max_range.at(2)-min_range.at(2))*z;

    ssm->addObstaclePosition(obs_location);

    obs = {x,y,z};

    // Select a random connection
    parent = sampler->sample();
    child  = sampler->sample();

    if((parent-child).norm()>1.0)
      child = parent+(child-parent)*dist(gen);

    connection = (child-parent);

    if(not checker->checkPath(parent,child))
      continue;

    // Joints velocity vector
    slowest_joint_time = (inv_max_speed.cwiseProduct(connection)).cwiseAbs().maxCoeff();
    dq = connection/slowest_joint_time;

    parent_scaled = (inv_q_limits).cwiseProduct(parent-lb);
    child_scaled  = (inv_q_limits).cwiseProduct(child -lb);

    dq_scaled = (inv_speed_limits).cwiseProduct(dq-min_speed);

    Sample sample;
    sample.parent = parent_scaled;
    sample.child = child_scaled;
    sample.dq = dq_scaled;
    sample.obstacle = obs;
    sample.scaling = ssm->computeScalingFactor(parent,child); //parent and child, not scaled!
    sample.scaling_first = ssm->computeScalingFactorAtQ(parent,dq,sample.speed_first,sample.distance_first,sample.v_safe_first);
    sample.scaling_mid = ssm->computeScalingFactorAtQ((child+parent)/2.0,dq,sample.speed_mid,sample.distance_mid,sample.v_safe_mid);
    sample.scaling_last = ssm->computeScalingFactorAtQ(child,dq,sample.speed_last,sample.distance_last,sample.v_safe_last);

    // Create a balanced dataset
    if(sample.scaling == 1.0)
    {
      if(v_scaling_1.size()<n_samples_per_vector)
        v_scaling_1.push_back(sample);
      else
        random_replace(v_scaling_1,sample);
    }
    else if(sample.scaling == std::numeric_limits<double>::infinity())
    {
      if(v_scaling_0.size()<n_samples_per_vector)
        v_scaling_0.push_back(sample);
      else
        random_replace(v_scaling_0,sample);
    }
    else if((1/sample.scaling)<1.0 && (1/sample.scaling)>=0.9)
    {
      if(v_scaling_09_1.size()<n_samples_per_vector)
        v_scaling_09_1.push_back(sample);
      else
        random_replace(v_scaling_09_1,sample);
    }
    else if((1/sample.scaling)<0.9 && (1/sample.scaling)>=0.8)
    {
      if(v_scaling_08_09.size()<n_samples_per_vector)
        v_scaling_08_09.push_back(sample);
      else
        random_replace(v_scaling_08_09,sample);
    }
    else if((1/sample.scaling)<0.8 && (1/sample.scaling)>=0.7)
    {
      if(v_scaling_07_08.size()<n_samples_per_vector)
        v_scaling_07_08.push_back(sample);
      else
        random_replace(v_scaling_07_08,sample);
    }
    else if((1/sample.scaling)<0.7 && (1/sample.scaling)>=0.6)
    {
      if(v_scaling_06_07.size()<n_samples_per_vector)
        v_scaling_06_07.push_back(sample);
      else
        random_replace(v_scaling_06_07,sample);
    }
    else if((1/sample.scaling)<0.6 && (1/sample.scaling)>=0.5)
    {
      if(v_scaling_05_06.size()<n_samples_per_vector)
        v_scaling_05_06.push_back(sample);
      else
        random_replace(v_scaling_05_06,sample);
    }
    else if((1/sample.scaling)<0.5 && (1/sample.scaling)>=0.4)
    {
      if(v_scaling_04_05.size()<n_samples_per_vector)
        v_scaling_04_05.push_back(sample);
      else
        random_replace(v_scaling_04_05,sample);
    }
    else if((1/sample.scaling)<0.4 && (1/sample.scaling)>=0.3)
    {
      if(v_scaling_03_04.size()<n_samples_per_vector)
        v_scaling_03_04.push_back(sample);
      else
        random_replace(v_scaling_03_04,sample);
    }
    else if((1/sample.scaling)<0.3 && (1/sample.scaling)>=0.2)
    {
      if(v_scaling_02_03.size()<n_samples_per_vector)
        v_scaling_02_03.push_back(sample);
      else
        random_replace(v_scaling_02_03,sample);
    }
    else if((1/sample.scaling)<0.2 && (1/sample.scaling)>=0.1)
    {
      if(v_scaling_01_02.size()<n_samples_per_vector)
        v_scaling_01_02.push_back(sample);
      else
        random_replace(v_scaling_01_02,sample);
    }
    else
    {
      if(v_scaling_0_01.size()<n_samples_per_vector)
        v_scaling_0_01.push_back(sample);
      else
        random_replace(v_scaling_0_01,sample);
    }

    vectors_fill[0]  = std::ceil(((double) (n_samples_per_vector,v_scaling_1    .size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[1]  = std::ceil(((double) (n_samples_per_vector,v_scaling_1    .size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[2]  = std::ceil(((double) (n_samples_per_vector,v_scaling_09_1 .size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[3]  = std::ceil(((double) (n_samples_per_vector,v_scaling_08_09.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[4]  = std::ceil(((double) (n_samples_per_vector,v_scaling_07_08.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[5]  = std::ceil(((double) (n_samples_per_vector,v_scaling_06_07.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[6]  = std::ceil(((double) (n_samples_per_vector,v_scaling_05_06.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[7]  = std::ceil(((double) (n_samples_per_vector,v_scaling_04_05.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[8]  = std::ceil(((double) (n_samples_per_vector,v_scaling_03_04.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[9]  = std::ceil(((double) (n_samples_per_vector,v_scaling_02_03.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[10] = std::ceil(((double) (n_samples_per_vector,v_scaling_01_02.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[11] = std::ceil(((double) (n_samples_per_vector,v_scaling_0_01 .size())/((double) n_samples_per_vector))*100.0);

    std::string output = "\r[=>";

    for(size_t i=0; i<vectors_fill.size();i++)
      output = output+" (v"+std::to_string(i)+" "+std::to_string(vectors_fill[i])+"%)";

    output = "\033[1;37;44m"+output+" <=]\033[0m";  //1->bold, 37->foreground white, 44->background blue

    if(v_scaling_1    .size()>=n_samples_per_vector && v_scaling_0_01 .size()>=n_samples_per_vector &&
       v_scaling_01_02.size()>=n_samples_per_vector && v_scaling_02_03.size()>=n_samples_per_vector &&
       v_scaling_03_04.size()>=n_samples_per_vector && v_scaling_04_05.size()>=n_samples_per_vector &&
       v_scaling_05_06.size()>=n_samples_per_vector && v_scaling_06_07.size()>=n_samples_per_vector &&
       v_scaling_07_08.size()>=n_samples_per_vector && v_scaling_08_09.size()>=n_samples_per_vector &&
       v_scaling_09_1 .size()>=n_samples_per_vector && v_scaling_0    .size()>=n_samples_per_vector)
    {
      output = output+"\033[1;5;32m Succesfully completed!\033[0m\n";
      std::cout<<output;

      break;
    }
    else
      std::cout<<output;
  }

  // Shuffle the vectors and extract the first n_samples_per_vector elements
  std::default_random_engine rng = std::default_random_engine {rd()};

  std::shuffle(std::begin(v_scaling_1    ), std::end(v_scaling_1    ), rng);
  std::shuffle(std::begin(v_scaling_0    ), std::end(v_scaling_0    ), rng);
  std::shuffle(std::begin(v_scaling_0_01 ), std::end(v_scaling_0_01 ), rng);
  std::shuffle(std::begin(v_scaling_01_02), std::end(v_scaling_01_02), rng);
  std::shuffle(std::begin(v_scaling_02_03), std::end(v_scaling_02_03), rng);
  std::shuffle(std::begin(v_scaling_03_04), std::end(v_scaling_03_04), rng);
  std::shuffle(std::begin(v_scaling_04_05), std::end(v_scaling_04_05), rng);
  std::shuffle(std::begin(v_scaling_05_06), std::end(v_scaling_05_06), rng);
  std::shuffle(std::begin(v_scaling_06_07), std::end(v_scaling_06_07), rng);
  std::shuffle(std::begin(v_scaling_07_08), std::end(v_scaling_07_08), rng);
  std::shuffle(std::begin(v_scaling_08_09), std::end(v_scaling_08_09), rng);
  std::shuffle(std::begin(v_scaling_09_1 ), std::end(v_scaling_09_1 ), rng);

  std::vector<Sample>::const_iterator first = v_scaling_1.begin();
  std::vector<Sample>::const_iterator last = v_scaling_1.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_1_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_1_vec.begin(),tmp_scaling_1_vec.end());

  first = v_scaling_0.begin();
  last = v_scaling_0.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_v_scaling_0(first, last);
  samples.insert(samples.end(),tmp_v_scaling_0.begin(),tmp_v_scaling_0.end());

  first = v_scaling_0_01.begin();
  last = v_scaling_0_01.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_01_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_01_vec.begin(),tmp_scaling_01_vec.end());

  first = v_scaling_01_02.begin();
  last = v_scaling_01_02.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_02_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_02_vec.begin(),tmp_scaling_02_vec.end());

  first = v_scaling_02_03.begin();
  last = v_scaling_02_03.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_03_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_03_vec.begin(),tmp_scaling_03_vec.end());

  first = v_scaling_03_04.begin();
  last = v_scaling_03_04.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_04_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_04_vec.begin(),tmp_scaling_04_vec.end());

  first = v_scaling_04_05.begin();
  last = v_scaling_04_05.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_05_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_05_vec.begin(),tmp_scaling_05_vec.end());

  first = v_scaling_05_06.begin();
  last = v_scaling_05_06.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_06_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_06_vec.begin(),tmp_scaling_06_vec.end());

  first = v_scaling_06_07.begin();
  last = v_scaling_06_07.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_07_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_07_vec.begin(),tmp_scaling_07_vec.end());

  first = v_scaling_07_08.begin();
  last = v_scaling_07_08.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_08_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_08_vec.begin(),tmp_scaling_08_vec.end());

  first = v_scaling_08_09.begin();
  last = v_scaling_08_09.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_09_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_09_vec.begin(),tmp_scaling_09_vec.end());

  first = v_scaling_09_1.begin();
  last = v_scaling_09_1.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_099_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_099_vec.begin(),tmp_scaling_099_vec.end());

  std::shuffle(std::begin(samples), std::end(samples), rng);

  // Save params in a database
  std::string path = ros::package::getPath("nn_ssm");
  path = path+"/scripts/data/";

  std::ofstream file_params;
  file_params.open(path+database_name+"_creation_params.bin", std::ios::out | std::ios::binary);

  const size_t bufsize = 1024 * 1024;
  std::unique_ptr<char[]> buf_params;
  buf_params.reset(new char[bufsize]);

  file_params.rdbuf()->pubsetbuf(buf_params.get(), bufsize);

  file_params.write((char*) &group_name       , sizeof(group_name                  ));
  file_params.write((char*) &base_frame       , sizeof(base_frame                  ));
  file_params.write((char*) &tool_frame       , sizeof(tool_frame                  ));
  file_params.write((char*) &max_cart_acc     , sizeof(max_cart_acc                ));
  file_params.write((char*) &t_r              , sizeof(t_r                         ));
  file_params.write((char*) &v_h              , sizeof(v_h                         ));
  file_params.write((char*) &n_objects        , sizeof(n_objects                   ));
  file_params.write((char*) &n_iter           , sizeof(n_iter                      ));
  file_params.write((char*) &min_safe_distance, sizeof(min_safe_distance           ));
  file_params.write((char*) &poi_names[0]     , sizeof(std::string)*poi_names.size());
  file_params.write((char*) &min_range[0]     , sizeof(double)     *min_range.size());
  file_params.write((char*) &max_range[0]     , sizeof(double)     *max_range.size());

  file_params.flush();
  file_params.close();

  // Save samples in a database
  std::ofstream file;
  file.open(path+database_name+".bin", std::ios::out | std::ios::binary);
  std::unique_ptr<char[]> buf;
  buf.reset(new char[bufsize]);

  file.rdbuf()->pubsetbuf(buf.get(), bufsize);

  double max_hr_distance = 0;
  double max_tang_speed = 0;
  double max_v_safe = 0;
  for(const Sample& sample:samples)
  {
    if(sample.distance_first>max_hr_distance)
      max_hr_distance = sample.distance_first;
    if(sample.distance_mid>max_hr_distance)
      max_hr_distance = sample.distance_mid;
    if(sample.distance_last>max_hr_distance)
      max_hr_distance = sample.distance_last;

    if(std::abs(sample.speed_first)>max_tang_speed)
      max_tang_speed = std::abs(sample.speed_first);
    if(std::abs(sample.speed_mid)>max_tang_speed)
      max_tang_speed = std::abs(sample.speed_mid);
    if(std::abs(sample.speed_last)>max_tang_speed)
      max_tang_speed = std::abs(sample.speed_last);

    if(sample.v_safe_first>max_v_safe)
      max_v_safe = sample.v_safe_first;
    if(sample.v_safe_mid>max_v_safe)
      max_v_safe = sample.v_safe_mid;
    if(sample.v_safe_last>max_v_safe)
      max_v_safe = sample.v_safe_last;
  }

  double min_tang_speed = -max_tang_speed;

  assert(max_hr_distance>0.0 && max_tang_speed>0.0);

  double max_scaling = 1000;
  for(Sample& sample:samples)
  {
    ROS_INFO("------------------------------");
    ROS_INFO_STREAM("sample: "<<sample);

    sample.distance_first = (sample.distance_first/max_hr_distance);
    sample.distance_mid   = (sample.distance_mid  /max_hr_distance);
    sample.distance_last  = (sample.distance_last /max_hr_distance);

    sample.speed_first = ((sample.speed_first-min_tang_speed)/(max_tang_speed-min_tang_speed));
    sample.speed_mid   = ((sample.speed_mid-min_tang_speed  )/(max_tang_speed-min_tang_speed));
    sample.speed_last  = ((sample.speed_last-min_tang_speed )/(max_tang_speed-min_tang_speed));

    sample.v_safe_first = (sample.v_safe_first/max_v_safe);
    sample.v_safe_mid   = (sample.v_safe_mid  /max_v_safe);
    sample.v_safe_last  = (sample.v_safe_last /max_v_safe);

    sample.scaling >= max_scaling?
          (sample.scaling = 0.0):
          (sample.scaling = 1.0/sample.scaling);

    sample.scaling_first >= max_scaling?
          (sample.scaling_first = 0.0):
          (sample.scaling_first = 1.0/sample.scaling_first);

    sample.scaling_mid >= max_scaling?
          (sample.scaling_mid = 0.0):
          (sample.scaling_mid = 1.0/sample.scaling_mid);

    sample.scaling_last >= max_scaling?
          (sample.scaling_last = 0.0):
          (sample.scaling_last = 1.0/sample.scaling_last);

    ROS_WARN_STREAM("sample: "<<sample);

    assert(sample.scaling       <=1 && sample.scaling       >=0);
    assert(sample.scaling_first <=1 && sample.scaling_first >=0);
    assert(sample.scaling_mid   <=1 && sample.scaling_mid   >=0);
    assert(sample.scaling_last  <=1 && sample.scaling_last  >=0);
    assert(sample.speed_first   <=1 && sample.speed_first   >=0);
    assert(sample.speed_mid     <=1 && sample.speed_mid     >=0);
    assert(sample.speed_last    <=1 && sample.speed_last    >=0);
    assert(sample.distance_first<=1 && sample.distance_first>=0);
    assert(sample.distance_mid  <=1 && sample.distance_mid  >=0);
    assert(sample.distance_last <=1 && sample.distance_last >=0);
    assert(sample.v_safe_first  <=1 && sample.v_safe_first  >=0);
    assert(sample.v_safe_mid    <=1 && sample.v_safe_mid    >=0);
    assert(sample.v_safe_last   <=1 && sample.v_safe_last   >=0);
  }

  std::vector<double> tmp;
  std::vector<double> sample_vector;
  for(const Sample& sample:samples)
  {
    sample_vector.clear();

    //parent
    tmp.clear();
    tmp.resize(sample.parent.size());
    Eigen::VectorXd::Map(&tmp[0], sample.parent.size()) = sample.parent;
    sample_vector.insert(sample_vector.end(),tmp.begin(),tmp.end());

    //child
    tmp.clear();
    tmp.resize(sample.child.size());
    Eigen::VectorXd::Map(&tmp[0], sample.child.size()) = sample.child;
    sample_vector.insert(sample_vector.end(),tmp.begin(),tmp.end());

    //obstacle
    sample_vector.insert(sample_vector.end(),sample.obstacle.begin(),sample.obstacle.end());

    //dq
    tmp.clear();
    tmp.resize(sample.dq.size());
    Eigen::VectorXd::Map(&tmp[0], sample.dq.size()) = sample.dq;
    sample_vector.insert(sample_vector.end(),tmp.begin(),tmp.end());

    //first/mid/last v_safe
    sample_vector.push_back(sample.v_safe_first);
    sample_vector.push_back(sample.v_safe_mid);
    sample_vector.push_back(sample.v_safe_last);

    //first/mid/last speed
    sample_vector.push_back(sample.speed_first);
    sample_vector.push_back(sample.speed_mid);
    sample_vector.push_back(sample.speed_last);

    //first/mid/last distance
    sample_vector.push_back(sample.distance_first);
    sample_vector.push_back(sample.distance_mid);
    sample_vector.push_back(sample.distance_last);

    // min/max scaling
    sample_vector.push_back(sample.scaling_first);
    sample_vector.push_back(sample.scaling_mid);
    sample_vector.push_back(sample.scaling_last);

    // scaling
    sample_vector.push_back(sample.scaling);

    file.write((char*)&sample_vector[0], sample_vector.size()*sizeof(double));
  }

  file.flush();
  file.close();

  ROS_INFO_STREAM("Dataset size -> "<<samples.size());

  return 0;
}
