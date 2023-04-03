cmake_minimum_required(VERSION 3.0.2)
project(nn_ssm)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

add_compile_options(-std=c++14)

# Locate the cmake file of torchlib
#list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/libtorch/share/cmake/Torch")
set(Torch_DIR "${CMAKE_CURRENT_LIST_DIR}/libtorch/share/cmake/Torch")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  ssm15066_estimators
)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp rospy ssm15066_estimators
  DEPENDS 
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Torch_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
src/neural_networks/feedforward_nn.cpp
src/ssm15066_estimators/ssm15066_estimatorNN.cpp
)
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${TORCH_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${TORCH_LIBRARIES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

add_executable(create_dataset src/create_dataset.cpp)
add_dependencies(create_dataset ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(create_dataset
 ${catkin_LIBRARIES}
)
