cmake_minimum_required(VERSION 3.0.2)
project(nn_ssm)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

add_compile_options(-std=c++17 -funroll-loops -Wall -Ofast)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  ssm15066_estimators
  rt_neural_wrapper
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp rospy ssm15066_estimators rt_neural_wrapper
  DEPENDS 
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
src/neural_networks/neural_network.cpp
src/ssm15066_estimators/ssm15066_estimatorNN.cpp
)
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

add_executable(create_dataset src/create_dataset.cpp)
add_dependencies(create_dataset ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(create_dataset
 ${catkin_LIBRARIES}
)

add_executable(create_dataset2 src/create_dataset2.cpp)
add_dependencies(create_dataset2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(create_dataset2
 ${catkin_LIBRARIES}
)

add_executable(create_dataset_connection src/create_dataset_connection.cpp)
add_dependencies(create_dataset_connection ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(create_dataset_connection
 ${catkin_LIBRARIES}
)

add_executable(nn_test src/neural_networks/nn_test.cpp)
add_dependencies(nn_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(nn_test
 ${catkin_LIBRARIES}
)

add_executable(nn_test_speed src/neural_networks/nn_test_speed.cpp)
add_dependencies(nn_test_speed ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(nn_test_speed
 ${catkin_LIBRARIES}
)

add_executable(compare_speed src/neural_networks/compare_speed.cpp)
add_dependencies(compare_speed ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(compare_speed
 ${catkin_LIBRARIES}
)
