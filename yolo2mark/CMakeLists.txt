cmake_minimum_required(VERSION 2.8.3)
project(yolo2mark)


find_package(catkin REQUIRED COMPONENTS
  darknet_ros_3d_msgs
  darknet_ros_msgs
  pcl_conversions
  pcl_ros
  roscpp
  roslint
  rospy
  sensor_msgs
  std_msgs
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
)

find_package(PCL REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES lesson_perception
  CATKIN_DEPENDS
    roscpp
    rospy
    sensor_msgs
    std_msgs
    message_runtime
    visualization_msgs
    tf
    pcl_ros
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(plane_segmatation src/perception_node.cpp)

target_link_libraries(plane_segmatation ${catkin_LIBRARIES} ${PCL_LIBRARIES})

