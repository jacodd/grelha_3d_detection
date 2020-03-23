#!/usr/bin/env python
# -*- coding: utf-8 -*-

from yolo_subscriber import YoloSubscriber, in_which_box, draw_box
import numpy as np
import cv2
import rospy
import argparse
import glob
import time
import math
import sensor_msgs.point_cloud2 as pc2

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA

if True:  # Add project root
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+'/'



from lib_draw_3d_joints import set_default_params
from lib_rgbd import RgbdImage, MyCameraInfo
from lib_ros_rgbd_pub_and_sub import ColorImageSubscriber, DepthImageSubscriber, CameraInfoSubscriber
from lib_ros_rgbd_pub_and_sub import ColorImagePublisher
from lib_geo_trans import rotx, roty, rotz, get_Rp_from_T, form_T
from o3d_bridge import *
# import open3d as o3d
''' ------------------------------ Settings ------------------------------------ '''
TOPIC_RES_IMAGE = "/3d_pointing/res_image"
DST_RES_IMAGE_VIZ = "output/res_img/"
YOLO_TOPIC_NAME = "darknet_ros/bounding_boxes"
# OBJECT_CLASSES = ["grelha"]
OBJECT_CLASSES = ["grelha","person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
DOT_RADIUS_2D_HIT_POINT = 8
# COLORS = np.random.randint(0, 255, size=(len(OBJECT_CLASSES), 3),	dtype="uint8")
COLORS = np.random.rand(len(OBJECT_CLASSES), 3)

''' ------------------------------ Command line inputs ------------------------------------ '''


def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description="Detect human joints and then draw in rviz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # -- Select data source.
    parser.add_argument(
        "-s", "--data_source",
        default="rostopic",
        choices=["rostopic", "disk"],
        help="The option `disk` is only for debug. "
        "Since darknet_ros(YOLO) package reads from rostopic, "
        "we need to set this as `rostopic` when not debugging.")
    parser.add_argument(
        "-y", "--detect_object", type=Bool,
        default=True,
        help="If this is True, you must have already started the object detection by "
        "`roslaunch ros_3d_pointing_detection darknet_ros.launch`. "
        "Set this to False when you are debugging.")
    parser.add_argument(
        "-z", "--detect_hand", type=Bool,
        default=False,
        help="The hand joints are not used here. Besides, it's very slow.")
    parser.add_argument(
        "-u", "--depth_unit", type=float,
        default="0.001",
        help="Depth is (pixel_value * depth_unit) meters "
        "at each pixel of the depth image.")
    parser.add_argument(
        "-r", "--is_using_realsense", type=Bool,
        default=True,
        help="If the data source is Realsense, set this to true. "
        "Then, the drawn joints will change the coordinate to be the same as "
        "Realsense's point cloud. The reason is,"
        "I used a different coordinate direction than Realsense."
        "(1) For me, I use X-Right, Y-Down, Z-Forward,"
        "which is the convention for camera."
        "(2) For Realsense ROS package, it's X-Forward, Y-Left, Z-Up.")

    # -- "rostopic" as data source.
    parser.add_argument("-a", "--ros_topic_color",
                        default="camera/color/image_raw")
    parser.add_argument("-b", "--ros_topic_depth",
                        default="/camera/depth/image_rect_raw")
    parser.add_argument("-c", "--ros_topic_camera_info",
                        default="camera/color/camera_info")

    # -- Get args.
    inputs = rospy.myargv()[1:]
    inputs = [s for s in inputs if s.replace(" ", "") != ""]  # Remove blanks.
    args = parser.parse_args(inputs)

    
    # -- Return
    return args


def Bool(v):
    ''' A bool class for argparser '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


''' ------------------------------ Data loader ------------------------------------ '''

class DataReader_ROS(object):
    def __init__(self, args):
        self._sub_c = ColorImageSubscriber(args.ros_topic_color)
        self._sub_d = DepthImageSubscriber(args.ros_topic_depth)
        self._sub_i = CameraInfoSubscriber(args.ros_topic_camera_info)
        self._depth_unit = args.depth_unit
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


''' ------------------------------ Math ------------------------------------ '''


def cam2world(xyz_in_camera, camera_pose):
    xyz_in_world = camera_pose.dot(np.append(xyz_in_camera, [1.0]))[0:3]
    return xyz_in_world


def cam2pixel(xyz_in_camera, camera_intrinsics):
    ''' Project a point represented in camera coordinate onto the image plane.
    Arguments:
        xyz_in_camera {np.ndarray}: (3, ).
        camera_intrinsics {np.ndarray}: 3x3.
    Return:
        xy {np.ndarray, np.float32}: (2, ). Column and row index.
    '''
    pt_3d_on_cam_plane = xyz_in_camera/xyz_in_camera[2]  # z=1
    xy = camera_intrinsics.dot(pt_3d_on_cam_plane)[0:2]
    xy = tuple(int(v) for v in xy)
    return xy

# def pixel2cam(uv, depth, camera_intrinsics):
#     d = depth[v,u]
            
#     z = d/depth_scale
    
#     x = (u - camera_intrinsics[0,2]) * z / camera_intrinsics[0,0]
#     y = (v - camera_intrinsics[1,2]) * z / camera_intrinsics[1,1]

#     return x,y,z

''' ------------------------------- Rviz 3D visualization ------------------------------------ '''



''' ------------------------------- 2D image drawer ------------------------------------- '''


def draw_2d_point(xyz, intrin_mat, img, color=[0, 0, 255]):
    xy1 = cam2pixel(xyz, intrin_mat)
    cv2.circle(img, xy1,
               radius=DOT_RADIUS_2D_HIT_POINT, color=color, thickness=-1)
    hit_point_2d = xy1
    return hit_point_2d

''' ------------------------------- Main ------------------------------------- '''


def main(args):
    # -- Data reader.
    data_reader = DataReader_ROS(args)

    # -- Result publisher.
    pub_res_img = ColorImagePublisher(TOPIC_RES_IMAGE)
    marker_publisher = rospy.Publisher('visualization_marker', MarkerArray)
    # YOLOv3.
    yolo_sub = YoloSubscriber(YOLO_TOPIC_NAME, OBJECT_CLASSES)

    # -- Camera pose (For Rviz visualization).
    cam_pose, cam_pose_pub = set_default_params()
    if args.is_using_realsense:  # Change coordinate.
        R, p = get_Rp_from_T(cam_pose)
        R = roty(math.pi/2).dot(rotz(-math.pi/2)).dot(R)
        cam_pose = form_T(R, p)

    
    while not rospy.is_shutdown():
        

        rgbd = data_reader.read_next_data()
        # t0 = time.time()
        pcl_xyz, _ = rgbd.create_point_cloud(depth_max=4.0)  # The point cloud.
        R = np.array([0, 1.57, -1.57])
        
        # print(pcl_xyz)
        color = data_reader.read_color()
        depth = data_reader.read_depth()
        intrin_mat = rgbd.intrinsic_matrix()

        depth_scale =float(1000)

        # Set camera pose only for visualizing 3d joints in Rviz.
        # (I tranform the ROS Markers' positions from camera frame to world frame,
        # and then publish them to ROS topic.)
        rgbd.set_camera_pose(cam_pose)

        marker_arr = []

        # Check if `hit_point_2d` is in any of
        #   the objects' bounding boxes detected by YOLOv3.
        
        bboxes = yolo_sub.get_bboxes()
        
        for i, bb in enumerate(bboxes):
            
            u = (bb[0] + bb[2])/2
            v = (bb[1] + bb[3])/2
            # print(u,v)
            # print(color.shape)
            # print(depth.shape)
            d = depth[v,u]
            
            xc = d/depth_scale
            yc = (u - intrin_mat[0,2]) * xc / intrin_mat[0,0]
            zc = (v - intrin_mat[1,2]) * xc / intrin_mat[1,1]
            
            sy = (((bb[2]) - intrin_mat[0,2]) * xc / intrin_mat[0,0])-(((bb[0]) - intrin_mat[0,2]) * xc / intrin_mat[0,0])
            sz = (((bb[3])- intrin_mat[1,2]) * xc / intrin_mat[1,1])-(((bb[1] )- intrin_mat[1,2]) * xc / intrin_mat[1,1])

            # print(Point(xc, yc, zc))
            marker = Marker(
                            type=1,
                            id=i,
                            lifetime=rospy.Duration(1.5),
                            pose=Pose(Point(0.99*xc , -yc, -zc), Quaternion(0, 0, 0, 1)),
                            scale=Vector3(0.33,sy, sz),
                            header=Header(frame_id='camera_link'),
                            color=ColorRGBA(1.0, 0.0, 0.0, 0.15),
                            # color=ColorRGBA(COLORS[bb[4]][0],COLORS[bb[4]][1],COLORS[bb[4]][2], 0.6),
                            
                            )
            marker_arr.append(marker)
            # marker_publisher.publish(marker)   
        # print(marker_arr) 
        marker_publisher.publish(marker_arr)     

        # -- Keep update camera pose for rviz visualization.
        cam_pose_pub.publish()
     



if __name__ == '__main__':
    node_name = "yolo2mark"
    rospy.init_node(node_name)
    # rospy.sleep(0.1)
    args = parse_command_line_arguments()
    main(args)
    rospy.logwarn("Node `{}` stops.".format(node_name))
