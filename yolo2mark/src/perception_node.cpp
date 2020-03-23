#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/PointCloud2.h> 

// PCL specific includes
#include <pcl_conversions/pcl_conversions.h>
#include "pcl_ros/transforms.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>

// Needed to return a Pose message
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>


class Perception
{
public:
    Perception(ros::NodeHandle* nodehandle); 

    
    ros::NodeHandle nh_; 
    ros::Subscriber pcl_sub_; 
    // ros::ServiceServer minimal_service_;
    ros::Publisher cropped_pub_;
    ros::Publisher table_pub_;
    ros::Publisher object_pub_;
    ros::Publisher cluster_pub_;

// private:    
    std::string cloud_topic, world_frame, camera_frame;
    float voxel_leaf_size;
    float x_filter_min, x_filter_max, y_filter_min, y_filter_max, z_filter_min, z_filter_max;
    int plane_max_iter;
    float plane_dist_thresh;
    float cluster_tol;
    int cluster_min_size;
    int cluster_max_size;
    
    void initializeSubscribers(); 
    void initializePublishers();
    // void initializeServices();
    
    void subscriberCallback(const sensor_msgs::PointCloud2::ConstPtr& pcl_msg); 
}; 

Perception::Perception(ros::NodeHandle* nodehandle):nh_(*nodehandle)
{   // constructor
    initializeSubscribers(); 
    initializePublishers();
    // initializeServices();
    
    //initialize variables here, as needed
     
    cloud_topic = nh_.param<std::string>("cloud_topic", "/camera/depht/points");
    world_frame = nh_.param<std::string>("world_frame", "camera_link");
    camera_frame = nh_.param<std::string>("camera_frame", "camera_link");
    voxel_leaf_size = nh_.param<float>("voxel_leaf_size", 0.002);
    x_filter_min = nh_.param<float>("x_filter_min", -5);
    x_filter_max = nh_.param<float>("x_filter_max",  5);
    y_filter_min = nh_.param<float>("y_filter_min", -5);
    y_filter_max = nh_.param<float>("y_filter_max",  5);
    z_filter_min = nh_.param<float>("z_filter_min", -5);
    z_filter_max = nh_.param<float>("z_filter_max",  5);
    plane_max_iter = nh_.param<int>("plane_max_iterations", 50);
    plane_dist_thresh = nh_.param<float>("plane_distance_threshold", 0.05);
    cluster_tol = nh_.param<float>("cluster_tolerance", 0.01);
    cluster_min_size = nh_.param<int>("cluster_min_size", 100);
    cluster_max_size = nh_.param<int>("cluster_max_size", 50000);

      
}

void Perception::initializeSubscribers()
{
    ROS_INFO("Initializing Subscribers");
    pcl_sub_ = nh_.subscribe("/camera/depth/color/points", 3, &Perception::subscriberCallback,this);

}

void Perception::initializePublishers()
{
    ROS_INFO("Initializing Publishers");
//    cropped_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("cropped_cloud", 1, true);
    table_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("table_cloud", 1, true);
    object_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("object_cluster", 1, true);
//    cluster_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("primary_cluster", 1, true);
    
}

void Perception::subscriberCallback(const sensor_msgs::PointCloud2::ConstPtr& pcl_msg)
{

//    // CONVERT POINTCLOUD ROS->PCL
//    pcl::PointCloud<pcl::PointXYZRGB> cloud;
//    pcl::fromROSMsg(*pcl_msg, cloud);


    // Voxel Grid Downsampling

    // Container for original & filtered data
    pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
    pcl::PCLPointCloud2 cloud_filtered;

    // Convert to PCL data type
    pcl_conversions::toPCL(*pcl_msg, *cloud);

    // Perform the actual filtering
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud (cloudPtr);
    sor.setLeafSize (0.0051, 0.0051, 0.0051);
    sor.filter (cloud_filtered);



    // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
    pcl::PointCloud<pcl::PointXYZRGB> cloudRansac;
    pcl::fromPCLPointCloud2 (cloud_filtered, cloudRansac);

    pcl::ModelCoefficients coefficients;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::PointCloud<pcl::PointXYZRGB>* cloud_plane = new pcl::PointCloud<pcl::PointXYZRGB>;
    pcl::PointCloud<pcl::PointXYZRGB>* cloud_f = new pcl::PointCloud<pcl::PointXYZRGB>;
    pcl::PointCloud<pcl::PointXYZRGB>* cloud_filter = new pcl::PointCloud<pcl::PointXYZRGB>;
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.00525);

    seg.setInputCloud (cloudRansac.makeShared ());
    seg.segment (*inliers, coefficients);
    if (inliers->indices.size () == 0)
    {
        ROS_WARN_STREAM ("Could not estimate a planar model for the given dataset.") ;
        //break;
    }

    // Extract inliers and outliers

    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud (cloudRansac.makeShared ());
    extract.setIndices(inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    ROS_INFO_STREAM("PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." );

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_f);
    // Publish ROS message
    // Convert to ROS data type
    sensor_msgs::PointCloud2 pc2_cloud;
    sensor_msgs::PointCloud2 pc2_cloud1;

//    pcl_conversions::fromPCL(*cloud_f, pc2_cloud);
    pcl::toROSMsg(*cloud_f, pc2_cloud);
    object_pub_.publish(pc2_cloud);
     pcl::toROSMsg(*cloud_plane, pc2_cloud1);
    table_pub_.publish(pc2_cloud1);
}

int main(int argc, char** argv) 
{
   // ROS set-ups:
    ros::init(argc, argv, "percetion_node"); 

    ros::NodeHandle nh;

    Perception perception(&nh);  

    ROS_INFO("spinning");
//    ros::Rate r(10);
//    while (ros::ok())
//    {
//        ros::spinOnce();
//        r.sleep();
//    }
    ros::spin();
    return 0;
} 