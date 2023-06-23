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

#define EIGEN_RUNTIME_NO_MALLOC // Define this symbol to enable runtime tests for allocations
#define EIGEN_STACK_ALLOCATION_LIMIT 262144
#include <Eigen/Dense>
#include <eigen3/Eigen/Eigen>
#include <ros/ros.h>


int main(int argc, char** argv)
{
  ros::init(argc, argv, "prova");
  ros::NodeHandle nh;

  ros::AsyncSpinner spinner(4);
  spinner.start();

  int n_neurons, batch_size, n_input;
  nh.getParam("n_neurons",n_neurons);
  nh.getParam("batch_size",batch_size);
  nh.getParam("n_input",n_input);

  // It's OK to allocate here
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> weight = Eigen::MatrixXd::Random(n_neurons, n_input);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> input  = Eigen::MatrixXd::Random(n_input, batch_size);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> output(n_neurons,batch_size);

  Eigen::internal::set_is_malloc_allowed(false);

  output.topLeftCorner(n_neurons,4).noalias() = weight*input.topLeftCorner(n_input,4);

  Eigen::internal::set_is_malloc_allowed(true);

  ROS_INFO_STREAM("output rows "<<output.rows()<<" cols "<<output.cols());
}

//int main(int argc, char** argv)
//{
//  uint N, M, P, max_cols;
//  N = 150;
//  M = 150;
//  P = 10;
//  max_cols = 50;


//  Eigen::MatrixXd B = Eigen::MatrixXd::Random(N, M);
//  Eigen::MatrixXd C = Eigen::MatrixXd::Random(M, max_cols);
//  Eigen::MatrixXd A(N,max_cols);

//  Eigen::internal::set_is_malloc_allowed(false);

//  A.block(0,0,N,P).noalias() = B*C.block(0,0,M,P);  // no assertion

//  Eigen::internal::set_is_malloc_allowed(true);

//  std::cout<<"first multiplication OK"<<std::endl;

//  N = 180;
//  M = 180;

//  Eigen::MatrixXd B2 = Eigen::MatrixXd::Random(N, M);
//  Eigen::MatrixXd C2 = Eigen::MatrixXd::Random(M, max_cols);
//  Eigen::MatrixXd A2(N,max_cols);

//  Eigen::internal::set_is_malloc_allowed(false);

//  A2.block(0,0,N,P).noalias() = B2*C2.block(0,0,M,P);  // assertion

//  Eigen::internal::set_is_malloc_allowed(true);

//  std::cout<<"second multiplication OK"<<std::endl;

//  return 0;
//}

