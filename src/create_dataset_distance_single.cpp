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
#include <min_distance_solvers/min_distance_solver.h>
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
  Eigen::VectorXd q;
  std::vector<double> obstacle;

  double distance,x,y,z;

  friend std::ostream& operator<<(std::ostream& os, const Sample& sample)
  {
    std::stringstream input, output;

    input<< "\n -- Input --\n q "<<sample.q.transpose();
    input<<"\n (x,y,z) obstacle "<<sample.obstacle[0]<<" "<<sample.obstacle[1]<<" "<<sample.obstacle[2];

    output<< "\n -- Output --\n distance: "<<sample.distance;
    output<< "\n x: "<<sample.x<<" y: "<<sample.y<<" z: "<<sample.z;

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
  ros::init(argc, argv, "create_ssm_database_distance");
  ros::NodeHandle nh;

  ros::AsyncSpinner spinner(4);
  spinner.start();

  std::srand(std::time(NULL));

  // Get params
  bool norm_1;
  nh.getParam("norm_1",norm_1);

  int n_objects;
  nh.getParam("n_objects",n_objects);

  int n_iter;
  nh.getParam("n_iter",n_iter);

  double max_distance;
  nh.getParam("max_distance",max_distance);

  std::string group_name;
  nh.getParam("group_name",group_name);

  std::string base_frame;
  nh.getParam("base_frame",base_frame);

  std::string tool_frame;
  nh.getParam("tool_frame",tool_frame);

  std::string database_name;
  nh.getParam("database_name",database_name);

  std::vector<std::string> poi_names;
  nh.getParam("poi_names",poi_names);

  std::vector<double> min_range;
  nh.getParam("min_range",min_range);

  std::vector<double> max_range;
  nh.getParam("max_range",max_range);

  database_name = database_name+"_"+std::to_string(int(n_iter/1000.0))+"k";

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
  Eigen::VectorXd inv_q_limits = (ub-lb).cwiseInverse();

  // Iterate over samples
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  ssm15066_estimator::MinDistanceSolverPtr dist_solver = std::make_shared<ssm15066_estimator::MinDistanceSolver>(chain);
  dist_solver->setPoiNames(poi_names);

  std::cout<<"Database "<<database_name<<" creation starts"<<std::endl;
  ros::Duration(5).sleep();

  double x, y, z;
  Eigen::Vector3d obs_location;
  Eigen::VectorXd q, q_scaled;

  std::vector<Sample> samples, v_distance_1, v_distance_0_01, v_distance_01_02, v_distance_02_03, v_distance_03_04,
      v_distance_04_05, v_distance_05_06, v_distance_06_07, v_distance_07_08, v_distance_08_09, v_distance_09_1;

  std::vector<double> obs;

  std::vector<int> vectors_fill(11,0);
  size_t n_samples_per_vector = (size_t)std::ceil(n_iter)/11;

  Eigen::VectorXd ones(lb.size());
  ones.setOnes();

  while(true && ros::ok())
  {
    // Create obstacle locations
    dist_solver->clearObstaclesPositions();

    x = dist(gen); obs_location[0] = min_range.at(0)+(max_range.at(0)-min_range.at(0))*x;
    y = dist(gen); obs_location[1] = min_range.at(1)+(max_range.at(1)-min_range.at(1))*y;
    z = dist(gen); obs_location[2] = min_range.at(2)+(max_range.at(2)-min_range.at(2))*z;

    dist_solver->addObstaclePosition(obs_location);

    if(norm_1)
    {
      // between -1 and 1
      x = x*2-1;
      y = y*2-1;
      z = z*2-1;
    }

    obs = {x,y,z};

    // Select a random connection
    q = sampler->sample();

    if(not checker->check(q))
      continue;

    q_scaled = (inv_q_limits).cwiseProduct(q-lb);

    assert([&]() ->bool{
             for(uint d=0;d<q_scaled.size();d++)
             {
               if(q_scaled[d]>1.0 || q_scaled[d]<0.0)
               return false;
             }
             return true;
           }());

    if(norm_1)
    {
      q_scaled = q_scaled*2 - ones;

      assert([&]() ->bool{
               for(uint d=0;d<q_scaled.size();d++)
               {
                 if(q_scaled[d]>1.0 || q_scaled[d]<-1.0)
                 return false;
               }
               return true;
             }());
    }

    ssm15066_estimator::DistancePtr distance = dist_solver->computeMinDistance(q);

    Sample sample;
    sample.q = q_scaled;
    sample.obstacle = obs;
    sample.distance = distance->distance_;
    sample.x = distance->poi_fk_[0];
    sample.y = distance->poi_fk_[1];
    sample.z = distance->poi_fk_[2];

    if(sample.distance > max_distance)
      sample.distance = max_distance;

    sample.distance = sample.distance/max_distance;

    assert(sample.distance>=0.0 && sample.distance<=1.0);

    // Create a balanced dataset
    if(sample.distance == 1.0)
    {
      if(v_distance_1.size()<n_samples_per_vector)
        v_distance_1.push_back(sample);
      else
        random_replace(v_distance_1,sample);
    }
    else if(sample.distance<1.0 && sample.distance>=0.9)
    {
      if(v_distance_09_1.size()<n_samples_per_vector)
        v_distance_09_1.push_back(sample);
      else
        random_replace(v_distance_09_1,sample);
    }
    else if(sample.distance<0.9 && sample.distance>=0.8)
    {
      if(v_distance_08_09.size()<n_samples_per_vector)
        v_distance_08_09.push_back(sample);
      else
        random_replace(v_distance_08_09,sample);
    }
    else if(sample.distance<0.8 && sample.distance>=0.7)
    {
      if(v_distance_07_08.size()<n_samples_per_vector)
        v_distance_07_08.push_back(sample);
      else
        random_replace(v_distance_07_08,sample);
    }
    else if(sample.distance<0.7 && sample.distance>=0.6)
    {
      if(v_distance_06_07.size()<n_samples_per_vector)
        v_distance_06_07.push_back(sample);
      else
        random_replace(v_distance_06_07,sample);
    }
    else if(sample.distance<0.6 && sample.distance>=0.5)
    {
      if(v_distance_05_06.size()<n_samples_per_vector)
        v_distance_05_06.push_back(sample);
      else
        random_replace(v_distance_05_06,sample);
    }
    else if(sample.distance<0.5 && sample.distance>=0.4)
    {
      if(v_distance_04_05.size()<n_samples_per_vector)
        v_distance_04_05.push_back(sample);
      else
        random_replace(v_distance_04_05,sample);
    }
    else if(sample.distance<0.4 && sample.distance>=0.3)
    {
      if(v_distance_03_04.size()<n_samples_per_vector)
        v_distance_03_04.push_back(sample);
      else
        random_replace(v_distance_03_04,sample);
    }
    else if(sample.distance<0.3 && sample.distance>=0.2)
    {
      if(v_distance_02_03.size()<n_samples_per_vector)
        v_distance_02_03.push_back(sample);
      else
        random_replace(v_distance_02_03,sample);
    }
    else if(sample.distance<0.2 && sample.distance>=0.1)
    {
      if(v_distance_01_02.size()<n_samples_per_vector)
        v_distance_01_02.push_back(sample);
      else
        random_replace(v_distance_01_02,sample);
    }
    else
    {
      if(v_distance_0_01.size()<n_samples_per_vector)
        v_distance_0_01.push_back(sample);
      else
        random_replace(v_distance_0_01,sample);
    }

    vectors_fill[0]  = std::ceil(((double) (n_samples_per_vector,v_distance_1    .size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[1]  = std::ceil(((double) (n_samples_per_vector,v_distance_09_1 .size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[2]  = std::ceil(((double) (n_samples_per_vector,v_distance_08_09.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[3]  = std::ceil(((double) (n_samples_per_vector,v_distance_07_08.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[4]  = std::ceil(((double) (n_samples_per_vector,v_distance_06_07.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[5]  = std::ceil(((double) (n_samples_per_vector,v_distance_05_06.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[6]  = std::ceil(((double) (n_samples_per_vector,v_distance_04_05.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[7]  = std::ceil(((double) (n_samples_per_vector,v_distance_03_04.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[8]  = std::ceil(((double) (n_samples_per_vector,v_distance_02_03.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[9]  = std::ceil(((double) (n_samples_per_vector,v_distance_01_02.size())/((double) n_samples_per_vector))*100.0);
    vectors_fill[10] = std::ceil(((double) (n_samples_per_vector,v_distance_0_01 .size())/((double) n_samples_per_vector))*100.0);

    std::string output = "\r[=>";

    for(size_t i=0; i<vectors_fill.size();i++)
      output = output+" (v"+std::to_string(i)+" "+std::to_string(vectors_fill[i])+"%)";

    output = "\033[1;37;44m"+output+" <=]\033[0m";  //1->bold, 37->foreground white, 44->background blue

    if(v_distance_1    .size()>=n_samples_per_vector && v_distance_0_01 .size()>=n_samples_per_vector &&
       v_distance_01_02.size()>=n_samples_per_vector && v_distance_02_03.size()>=n_samples_per_vector &&
       v_distance_03_04.size()>=n_samples_per_vector && v_distance_04_05.size()>=n_samples_per_vector &&
       v_distance_05_06.size()>=n_samples_per_vector && v_distance_06_07.size()>=n_samples_per_vector &&
       v_distance_07_08.size()>=n_samples_per_vector && v_distance_08_09.size()>=n_samples_per_vector &&
       v_distance_09_1 .size()>=n_samples_per_vector)
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

  std::shuffle(std::begin(v_distance_1    ), std::end(v_distance_1    ), rng);
  std::shuffle(std::begin(v_distance_0_01 ), std::end(v_distance_0_01 ), rng);
  std::shuffle(std::begin(v_distance_01_02), std::end(v_distance_01_02), rng);
  std::shuffle(std::begin(v_distance_02_03), std::end(v_distance_02_03), rng);
  std::shuffle(std::begin(v_distance_03_04), std::end(v_distance_03_04), rng);
  std::shuffle(std::begin(v_distance_04_05), std::end(v_distance_04_05), rng);
  std::shuffle(std::begin(v_distance_05_06), std::end(v_distance_05_06), rng);
  std::shuffle(std::begin(v_distance_06_07), std::end(v_distance_06_07), rng);
  std::shuffle(std::begin(v_distance_07_08), std::end(v_distance_07_08), rng);
  std::shuffle(std::begin(v_distance_08_09), std::end(v_distance_08_09), rng);
  std::shuffle(std::begin(v_distance_09_1 ), std::end(v_distance_09_1 ), rng);

  std::vector<Sample>::const_iterator first = v_distance_1.begin();
  std::vector<Sample>::const_iterator last = v_distance_1.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_1_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_1_vec.begin(),tmp_scaling_1_vec.end());

  first = v_distance_0_01.begin();
  last = v_distance_0_01.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_01_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_01_vec.begin(),tmp_scaling_01_vec.end());

  first = v_distance_01_02.begin();
  last = v_distance_01_02.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_02_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_02_vec.begin(),tmp_scaling_02_vec.end());

  first = v_distance_02_03.begin();
  last = v_distance_02_03.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_03_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_03_vec.begin(),tmp_scaling_03_vec.end());

  first = v_distance_03_04.begin();
  last = v_distance_03_04.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_04_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_04_vec.begin(),tmp_scaling_04_vec.end());

  first = v_distance_04_05.begin();
  last = v_distance_04_05.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_05_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_05_vec.begin(),tmp_scaling_05_vec.end());

  first = v_distance_05_06.begin();
  last = v_distance_05_06.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_06_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_06_vec.begin(),tmp_scaling_06_vec.end());

  first = v_distance_06_07.begin();
  last = v_distance_06_07.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_07_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_07_vec.begin(),tmp_scaling_07_vec.end());

  first = v_distance_07_08.begin();
  last = v_distance_07_08.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_08_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_08_vec.begin(),tmp_scaling_08_vec.end());

  first = v_distance_08_09.begin();
  last = v_distance_08_09.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_09_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_09_vec.begin(),tmp_scaling_09_vec.end());

  first = v_distance_09_1.begin();
  last = v_distance_09_1.begin() + n_samples_per_vector;
  std::vector<Sample> tmp_scaling_099_vec(first, last);
  samples.insert(samples.end(),tmp_scaling_099_vec.begin(),tmp_scaling_099_vec.end());

  std::shuffle(std::begin(samples), std::end(samples), rng);

  std::string path = ros::package::getPath("nn_ssm");
  path = path+"/scripts/data/";

  std::ofstream file;
  file.open(path+database_name+".bin", std::ios::out | std::ios::binary);
  const size_t bufsize = 1024 * 1024;
  std::unique_ptr<char[]> buf;
  buf.reset(new char[bufsize]);

  file.rdbuf()->pubsetbuf(buf.get(), bufsize);

  double max_x, max_y, max_z;
  max_x = 0.0; max_y = 0.0; max_z = 0.0;
  for(Sample& s:samples)
  {
    if(std::abs(s.x)>max_x)
      max_x = std::abs(s.x);
    if(std::abs(s.y)>max_y)
      max_y = std::abs(s.y);
    if(std::abs(s.z)>max_z)
      max_z = std::abs(s.z);
  }


  std::vector<double> tmp;
  std::vector<double> sample_vector;
  for(Sample& sample:samples)
  {
    sample_vector.clear();

    sample.x = (sample.x+max_x)/(2*max_x);
    sample.y = (sample.y+max_y)/(2*max_y);
    sample.z = (sample.z+max_z)/(2*max_z);

    if(max_x == 0)
      throw std::runtime_error("x");

    ROS_INFO_STREAM("sample" << sample);

    //q
    tmp.clear();
    tmp.resize(sample.q.size());
    Eigen::VectorXd::Map(&tmp[0], sample.q.size()) = sample.q;
    sample_vector.insert(sample_vector.end(),tmp.begin(),tmp.end());

    assert([&]() ->bool{
             if(norm_1)
             {
               for(uint d=0;d<sample.q.size();d++)
               {
                 if(sample.q[d]>1.0 || sample.q[d]<-1.0)
                 return false;
               }
             }
             else
             {
               for(uint d=0;d<sample.q.size();d++)
               {
                 if(sample.q[d]>1.0 || sample.q[d]<0.0)
                 return false;
               }
             }
             return true;
           }());

    //obstacle
    sample_vector.insert(sample_vector.end(),sample.obstacle.begin(),sample.obstacle.end());

    assert([&]() ->bool{
             if(norm_1)
             {
               for(uint d=0;d<sample.obstacle.size();d++)
               {
                 if(sample.obstacle[d]>1.0 || sample.obstacle[d]<-1.0)
                 return false;
               }
             }
             else
             {
               for(uint d=0;d<sample.q.size();d++)
               {
                 if(sample.obstacle[d]>1.0 || sample.obstacle[d]<0.0)
                 return false;
               }
             }
             return true;
           }());

    //fk x,y,z
    sample_vector.push_back(sample.x);
    sample_vector.push_back(sample.y);
    sample_vector.push_back(sample.z);

    assert([&]() ->bool{

             if(sample.x<0.0 || sample.x>1.0)
             return false;

             if(sample.y<0.0 || sample.y>1.0)
             return false;

             if(sample.z<0.0 || sample.z>1.0)
             return false;

             return true;
           }());

    // distance
    sample_vector.push_back(sample.distance);

    assert([&]() ->bool{

             if(sample.distance>1.0 || sample.distance<0.0)
             return false;

             return true;
           }());

    file.write((char*)&sample_vector[0], sample_vector.size()*sizeof(double));
  }

  file.flush();
  file.close();

  ROS_INFO_STREAM("Dataset size -> "<<samples.size());

  return 0;
}
