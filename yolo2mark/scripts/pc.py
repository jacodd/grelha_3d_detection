#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import rospy
import argparse
import glob
import time
import math
import sensor_msgs.point_cloud2 as pc2
from lib_rgbd import RgbdImage, MyCameraInfo
from lib_ros_rgbd_pub_and_sub import ColorImageSubscriber, DepthImageSubscriber, CameraInfoSubscriber
from lib_geo_trans import rotx, roty, rotz, get_Rp_from_T, form_T
from o3d_bridge import *


''' ------------------------------ Data loader ------------------------------------ '''

class DataReader_ROS(object):
    def __init__(self ):
        self._sub_c = ColorImageSubscriber('/camera/color/image_raw')
        self._sub_d = DepthImageSubscriber('/camera/depth/image_rect_raw')
        self._sub_i = CameraInfoSubscriber('/camera/color/camera_info')
        self._depth_unit = 0.001
        self._camera_info = None
        self._cnt_imgs = 0

    def read_next_data(self):
        
        depth = self.read_depth()
        color = self.read_color()
        camera_info = self.get_camera_info()
        self._cnt_imgs += 1
        rgbd = RgbdImage(color, depth,
                         camera_info,
                         depth_unit=self._depth_unit)
        return rgbd

    def get_camera_info(self):
        '''
        Since a camera parameter won't change (usually),
        we read it from cache after it's initialized.
        '''
        if self._camera_info is None:
            while (not self._sub_i.has_camera_info()) and (not rospy.is_shutdown):
                rospy.sleep(0.001)
            if self._sub_i.has_camera_info:
                self._camera_info = MyCameraInfo(
                    ros_camera_info=self._sub_i.get_camera_info())
        return self._camera_info

    def read_depth(self):
        while not self._sub_d.has_image() and (not rospy.is_shutdown()):
            rospy.sleep(0.001)
        depth = self._sub_d.get_image()
        return depth

    def read_color(self):
        while not self._sub_c.has_image() and (not rospy.is_shutdown()):
            rospy.sleep(0.001)
        color = self._sub_c.get_image()
        return color



''' ------------------------------- plane seg ------------------------------------- '''


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh    

''' ------------------------------- Main ------------------------------------- '''


def main():
    # -- Data reader.
    data_reader = DataReader_ROS()

    # -- Publisher.
    cloud_pub = rospy.Publisher('cloud', pc2.PointCloud2,queue_size=5)
    seg_cloud_pub = rospy.Publisher('seg_cloud', pc2.PointCloud2,queue_size=5)
   
    
    while not rospy.is_shutdown():
        

        rgbd = data_reader.read_next_data()
        pcl_xyz, open3d_cloud = rgbd.create_point_cloud(depth_max=4.0)  # The point cloud.
        R = np.array([0, 1.57, -1.57]) #to camera axis
        open3d_cloud = o3d.geometry.PointCloud.rotate(open3d_cloud,R,center=False)
        open3d_cloud = o3d.geometry.PointCloud.crop(open3d_cloud, [-1.5,-1.5,-1.5], [12,1,1.5])
        cloud_pub.publish(convertCloudFromOpen3dToRos(open3d_cloud))



     



if __name__ == '__main__':
    node_name = "seg_pc"
    rospy.init_node(node_name)
    # rospy.sleep(0.1)
    main()
    rospy.logwarn("Node `{}` stops.".format(node_name))
