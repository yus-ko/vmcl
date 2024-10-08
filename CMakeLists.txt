cmake_minimum_required(VERSION 3.0.2)
project(vmcl)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  message_generation
  cv_bridge
  message_filters
  geometry_msgs
  std_msgs
  tf
  tf2
  tf2_ros
  nav_msgs
  pcl_ros
  dynamic_reconfigure
  potbot_lib
  potbot_msgs
)

find_package(OpenCV REQUIRED)
find_package(PkgConfig)
# pkg_check_modules(RealSense2 REQUIRED realsense2)
pkg_check_modules(OpenCV REQUIRED opencv)
# include_directories(${RealSense2_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
# link_directories(${RealSense2_LIBRARY_DIRS} ${OpenCV_LIBRARY_DIRS})

generate_dynamic_reconfigure_options(
  cfg/VMCL.cfg
)

generate_messages(
  DEPENDENCIES
  std_msgs 
  geometry_msgs 
  sensor_msgs
  nav_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES vmcl
  CATKIN_DEPENDS roscpp rospy sensor_msgs message_runtime std_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_definitions(-std=c++14)

add_executable(vmcl src/vmcl.cpp)
add_dependencies(vmcl ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(vmcl ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(marker_broadcaster src/broadcast.cpp)
target_link_libraries(marker_broadcaster ${catkin_LIBRARIES})

add_executable(sensor_filter src/sensor_filter.cpp)
target_link_libraries(sensor_filter ${catkin_LIBRARIES})

add_library(${PROJECT_NAME}_lib src/particle.cpp)
target_link_libraries(${PROJECT_NAME}_lib ${catkin_LIBRARIES})