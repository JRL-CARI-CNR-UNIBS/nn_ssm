#include <nn_ssm/neural_networks/neural_network.h>
#include <random>
#include <ros/ros.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "test_nn");
  ros::AsyncSpinner spinner(4);
  spinner.start();

  ros::NodeHandle nh;

  std::string name;
  if(not nh.getParam("namespace",name))
  {
    ROS_ERROR("namespace not properly defined");
    return 1;
  }

  int n_samples;
  if(not nh.getParam("n_samples",n_samples))
    n_samples = 1;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  Eigen::MatrixXd input = Eigen::MatrixXd::NullaryExpr(15,n_samples,[&](){return dis(gen);});

  neural_network::NeuralNetwork nn;

  double time;
  ROS_INFO("Importing neural network");
  ros::WallTime tic = ros::WallTime::now();
  nn.importFromParam(nh,name);
  time = (ros::WallTime::now()-tic).toSec();
  ROS_INFO_STREAM("Imported -> "<<time);

  ROS_INFO_STREAM("NN "<<nn);

  tic = ros::WallTime::now();
  neural_network::MatrixXn out = nn.forward(input);
  time = (ros::WallTime::now()-tic).toSec();

  ROS_INFO_STREAM("Time with "<<n_samples<<" samples -> "<<time);
  ROS_INFO_STREAM("Time for 1 sample -> "<<time/n_samples);

  return 0;
}

