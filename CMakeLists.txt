cmake_minimum_required(VERSION 3.0.2)
project(nn_ssm)
add_compile_options(-std=c++17)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

find_package(catkin REQUIRED COMPONENTS
  ssm15066_estimators
  roscpp
  rospy
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES nn_ssm
  CATKIN_DEPENDS ssm15066_estimators roscpp rospy
  DEPENDS 
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(create_dataset src/create_dataset.cpp)
add_dependencies(create_dataset ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(create_dataset
 ${catkin_LIBRARIES}
)
