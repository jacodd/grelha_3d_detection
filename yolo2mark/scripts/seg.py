#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy as np
import cv2
import rospy
import argparse
import glob
import time
import math

import open3d as o3d

from o3d_bridge import *


class DataReader_ROS(object):
    def __init__(self):
        x = 0 

def main():
    data_reader = DataReader_ROS()
    sub = rospy.Subscriber(/cloud)


if __name__ == '__main__':
    node_name = "detect_and_draw_joints"
    rospy.init_node(node_name)
    main()
    rospy.logwarn("Node `{}` stops.".format(node_name))