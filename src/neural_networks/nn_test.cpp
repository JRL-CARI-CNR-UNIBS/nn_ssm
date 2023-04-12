#include <ros/ros.h>
#include <nn_ssm/neural_networks/neural_network.h>

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
  std::vector<double> sample1, sample2, samples;
  if(not nh.getParam("sample1",sample1) || not nh.getParam("sample2",sample2))
  {
    ROS_ERROR("samples not properly defined");
    return 1;
  }
  assert(sample1.size() == sample2.size());

  samples = sample1;
  samples.insert(samples.end(),sample2.begin(),sample2.end());

  Eigen::Map<neural_network::MatrixXn> input(samples.data(),sample1.size(),2);

  assert(input.cols() == 2);
  assert(input.rows() == sample1.size());

  neural_network::NeuralNetwork nn;
  nn.importFromParam(nh,name);

  neural_network::MatrixXn out = nn.forward(input);

  ROS_INFO_STREAM("output 0 -> "<<out.col(0).transpose());
  ROS_INFO_STREAM("output 1 -> "<<out.col(1).transpose());

  return 0;
}

