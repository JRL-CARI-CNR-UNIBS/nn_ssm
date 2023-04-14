#include <random>
#include <nn_ssm/neural_networks/neural_network.h>
#include <ros/ros.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <length_penalty_metrics.h>
#include <ssm15066_estimators/parallel_ssm15066_estimator2D.h>
#include <thread-pool/BS_thread_pool.hpp>
#include <graph_core/informed_sampler.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "test_compare_speed_nn");
  ros::NodeHandle nh;

  ros::AsyncSpinner aspin(4);
  aspin.start();

  srand((unsigned int)time(NULL));

  std::string base_frame;
  nh.getParam("base_frame",base_frame);

  std::string tool_frame;
  nh.getParam("tool_frame",tool_frame);

  std::string group_name;
  nh.getParam("group_name",group_name);

  std::string object_type;
  nh.getParam("object_type",object_type);

  std::vector<std::string> poi_names;
  nh.getParam("poi_names",poi_names);

  double max_cart_acc;
  nh.getParam("max_cart_acc",max_cart_acc);

  double tr;
  nh.getParam("Tr",tr);

  double min_distance;
  nh.getParam("min_distance",min_distance);

  double v_h;
  nh.getParam("v_h",v_h);

  double max_step_size;
  nh.getParam("ssm_max_step_size",max_step_size);

  double max_distance;
  nh.getParam("max_distance",max_distance);

  int n_tests;
  nh.getParam("n_tests",n_tests);

  int n_object;
  nh.getParam("n_object",n_object);

  int n_threads;
  nh.getParam("ssm_n_threads",n_threads);

  std::string name;
  if(not nh.getParam("namespace",name))
  {
    ROS_ERROR("namespace not properly defined");
    return 1;
  }

  moveit::planning_interface::MoveGroupInterface move_group(group_name);
  robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
  robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();

  std::vector<std::string> joint_names = kinematic_model->getJointModelGroup(group_name)->getActiveJointModelNames();

  unsigned int dof = joint_names.size();
  Eigen::VectorXd lb(dof);
  Eigen::VectorXd ub(dof);

  for (unsigned int idx = 0; idx < dof; idx++)
  {
    const robot_model::VariableBounds& bounds = kinematic_model->getVariableBounds(joint_names.at(idx));
    if (bounds.position_bounded_)
    {
      lb(idx) = bounds.min_position_;
      ub(idx) = bounds.max_position_;
    }
  }

  Eigen::Vector3d grav; grav << 0, 0, -9.806;
  rosdyn::ChainPtr chain = rosdyn::createChain(*robot_model_loader.getURDF(),base_frame,tool_frame,grav);
  ssm15066_estimator::SSM15066Estimator2DPtr ssm = std::make_shared<ssm15066_estimator::SSM15066Estimator2D>(chain,max_step_size);
  ssm15066_estimator::ParallelSSM15066Estimator2DPtr parallel_ssm = std::make_shared<ssm15066_estimator::ParallelSSM15066Estimator2D>(chain,max_step_size,n_threads);

  ssm->setHumanVelocity(v_h,false);
  ssm->setMaxCartAcc(max_cart_acc,false);
  ssm->setReactionTime(tr,false);
  ssm->setMinDistance(min_distance,false);
  ssm->setPoiNames(poi_names);
  ssm->updateMembers();

  parallel_ssm->setHumanVelocity(v_h,false);
  parallel_ssm->setMaxCartAcc(max_cart_acc,false);
  parallel_ssm->setReactionTime(tr,false);
  parallel_ssm->setMinDistance(min_distance,false);
  parallel_ssm->setPoiNames(poi_names);
  parallel_ssm->updateMembers();

  neural_network::NeuralNetwork nn;

  ROS_INFO("Importing neural network");
  ros::WallTime tic = ros::WallTime::now();
  nn.importFromParam(nh,name);
  ROS_INFO_STREAM("Imported -> "<<(ros::WallTime::now()-tic).toSec());

  Eigen::Vector3d obstacle_position;

  std::random_device rseed;
  std::mt19937 gen(rseed());
  std::uniform_real_distribution<double> dist(-1.5,1.5);
  std::uniform_real_distribution<double> dist_z(0.0,1.5);

  std::vector<Eigen::VectorXd> obs_norm_vector;

  for(unsigned int i=0;i<n_object;i++)
  {
    obstacle_position[0] = dist(gen);
    obstacle_position[1] = dist(gen);
    obstacle_position[2] = dist_z(gen);

    ssm->addObstaclePosition(obstacle_position);
    parallel_ssm->addObstaclePosition(obstacle_position);

    Eigen::Vector3d obs_norm;
    obs_norm << ((obstacle_position[0]+1.5)/3.0),((obstacle_position[1]+1.5)/3.0),((obstacle_position[2])/1.5);
    obs_norm_vector.push_back(obs_norm);
  }

  pathplan::SamplerPtr sampler = std::make_shared<pathplan::InformedSampler>(lb, ub, lb, ub);

  double distance;
  Eigen::VectorXd parent, child;
  std::vector<double> time_euclidean, time_ssm, time_parallel_ssm, time_nn;

  bool progress_bar_full = false;
  unsigned int progress = 0;

  ros::Duration(5).sleep();

  for(unsigned int i=0;i<n_tests;i++)
  {
    parent = sampler->sample();
    child  = sampler->sample();

    distance = (child-parent).norm();

    if(distance>max_distance || distance<0.1)
      child = parent+(child-parent)*max_distance/distance;

    neural_network::MatrixXn input_matrix(15,n_object);
    for(uint n=0;n<n_object;n++)
    {
      Eigen::VectorXd input(parent.size()+child.size()+3);
      input<<parent,child,obs_norm_vector.at(n);
      input_matrix.col(n) = input;
    }

    tic = ros::WallTime::now();
    (parent-child).norm();
    time_euclidean.push_back((ros::WallTime::now()-tic).toSec());

    tic = ros::WallTime::now();
    nn.forward(input_matrix);
    time_nn.push_back((ros::WallTime::now()-tic).toSec());

    tic = ros::WallTime::now();
    ssm->computeScalingFactor(parent,child);
    time_ssm.push_back((ros::WallTime::now()-tic).toSec());

    tic = ros::WallTime::now();
    parallel_ssm->computeScalingFactor(parent,child);
    time_parallel_ssm.push_back((ros::WallTime::now()-tic).toSec());

    progress = std::ceil(((double)(i+1.0))/((double)n_tests)*100.0);
    if(progress%5 == 0 && not progress_bar_full)
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

  //Mean
  double sum_time_nn           = std::accumulate(time_nn          .begin(),time_nn          .end(),0.0);
  double sum_time_ssm          = std::accumulate(time_ssm         .begin(),time_ssm         .end(),0.0);
  double sum_time_euclidean    = std::accumulate(time_euclidean   .begin(),time_euclidean   .end(),0.0);
  double sum_time_parallel_ssm = std::accumulate(time_parallel_ssm.begin(),time_parallel_ssm.end(),0.0);

  double mean_time_nn           = sum_time_nn          /time_nn          .size();
  double mean_time_ssm          = sum_time_ssm         /time_ssm         .size();
  double mean_time_euclidean    = sum_time_euclidean   /time_euclidean   .size();
  double mean_time_parallel_ssm = sum_time_parallel_ssm/time_parallel_ssm.size();

  //Stdev
  double accum = 0.0;
  std::for_each (std::begin(time_nn), std::end(time_nn), [&](const double d) {
      accum += (d - mean_time_nn) * (d - mean_time_nn);
  });

  double stdev_time_nn = sqrt(accum/(time_nn.size()-1));

  accum = 0.0;
  std::for_each (std::begin(time_ssm), std::end(time_ssm), [&](const double d) {
      accum += (d - mean_time_ssm) * (d - mean_time_ssm);
  });

  double stdev_time_ssm = sqrt(accum/(time_ssm.size()-1));

  accum = 0.0;
  std::for_each (std::begin(time_euclidean), std::end(time_euclidean), [&](const double d) {
      accum += (d - mean_time_euclidean) * (d - mean_time_euclidean);
  });

  double stdev_time_euclidean = sqrt(accum/(time_euclidean.size()-1));

  accum = 0.0;
  std::for_each (std::begin(time_parallel_ssm), std::end(time_parallel_ssm), [&](const double d) {
      accum += (d - mean_time_parallel_ssm) * (d - mean_time_parallel_ssm);
  });

  double stdev_time_parallel_ssm = sqrt(accum/(time_parallel_ssm.size()-1));

  ROS_BOLDCYAN_STREAM ("Mean nn: "              <<mean_time_nn           <<" s"<<" stdev "<<stdev_time_nn         );
  ROS_BOLDWHITE_STREAM("Mean Euclidean metric: "<<mean_time_euclidean    <<" s"<<" stdev "<<stdev_time_euclidean  );
  ROS_BOLDCYAN_STREAM ("Mean ssm: "             <<mean_time_ssm          <<" s"<<" stdev "<<stdev_time_ssm         );
  ROS_BOLDGREEN_STREAM("Mean parallel ssm: "    <<mean_time_parallel_ssm <<" s"<<" stdev "<<stdev_time_parallel_ssm);

  ROS_BOLDRED_STREAM("nn is "<<(mean_time_nn/mean_time_euclidean)   <<" time slower than Euclidean metrics");
  ROS_BOLDRED_STREAM("nn is "<<(mean_time_nn/mean_time_ssm)         <<" time slower than ssm"              );
  ROS_BOLDRED_STREAM("nn is "<<(mean_time_nn/mean_time_parallel_ssm)<<" time slower than parallel ssm"     );

  return 0;
}
