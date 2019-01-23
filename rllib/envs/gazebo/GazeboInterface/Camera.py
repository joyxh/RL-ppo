#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '20/11/2017'
__copyright__ = "RR"
__all__ = [
    'Camera',
]

import rospy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
import time
import sys
import os

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))


class Camera(object):
    def __init__(self, rgbImageTopic=None, depthImageTopic=None, pointCloudTopic=None):
        """

        :param rgbImageTopic:
        :param depthImageTopic:
        :param pointCloudTopic:
        """
        object.__init__(self)
        assert isinstance(rgbImageTopic, str) or rgbImageTopic is None
        assert isinstance(depthImageTopic, str) or depthImageTopic is None
        assert isinstance(pointCloudTopic, str) or pointCloudTopic is None
        self.__rgb_image_topic = rgbImageTopic
        self.__depth_image_topic = depthImageTopic
        self.__point_cloud_topic = pointCloudTopic
        if rospy.get_node_uri() is None:
            rospy.init_node("GazeboInterface",
                            log_level=rospy.INFO,
                            argv=rospy.myargv(argv=sys.argv),
                            anonymous=True)
            rospy.loginfo("Camera: Ros node init.")
        self.__cvBridge = CvBridge()
        if self.__rgb_image_topic is not None:
            self.__rgb_image = None
            self.__write_rgb_flag = False
            rospy.loginfo("Initial ros rgb image topic '%s'." % (self.__rgb_image_topic,))
            rospy.Subscriber(self.__rgb_image_topic, Image, self.__rgbCallback)

        if self.__depth_image_topic is not None:
            self.__depth_image = None
            self.__write_depth_flag = False
            rospy.loginfo("Initial ros depth image topic '%s'." % (self.__depth_image_topic,))
            rospy.Subscriber(self.__depth_image_topic, Image, self.__depthCallback)

        if self.__point_cloud_topic is not None:
            self.__point_cloud = None
            self.__write_point_flag = False
            rospy.loginfo("Initial ros point cloud topic '%s'." % (self.__point_cloud_topic,))
            rospy.Subscriber(self.__point_cloud_topic, PointCloud2, self.__pointCallback)

    def __rgbCallback(self, rgb):
        try:
            self.__rgb_image = self.__cvBridge.imgmsg_to_cv2(rgb, "bgr8")
            self.__write_rgb_flag = True
        except CvBridgeError as e:
            print(e)

    def __depthCallback(self, depth):
        try:
            self.__depth_image = self.__cvBridge.imgmsg_to_cv2(depth, "passthrough")
            self.__write_depth_flag = True
        except CvBridgeError as e:
            print(e)

    def __pointCallback(self, point):
        self.__point_cloud = pc2.read_points(point, skip_nans=True)
        self.__write_point_flag = True

    def getRGBImage(self):
        """

        :return: cv mat format image data
        """
        assert self.__rgb_image_topic is not None, "No initial rgb image topic."
        self.__write_rgb_flag = False
        startTime = time.time()
        while not self.__write_rgb_flag:
            if time.time() - startTime > 2.0:
                return None
        return self.__rgb_image

    def getDepthImage(self):
        """

        :return: cv mat format image data
        """
        assert self.__depth_image_topic is not None, "No initial rgb image topic."
        self.__write_depth_flag = False
        startTime = time.time()
        while not self.__write_depth_flag:
            if time.time() - startTime > 2.0:
                return None
        return self.__depth_image

    def getPointCloud(self):
        """

        :return: point cloud 2 msg
        """
        assert self.__point_cloud_topic is not None, "No initial point cloud topic."
        self.__write_point_flag = False
        startTime = time.time()
        while not self.__write_point_flag:
            if time.time() - startTime > 2.0:
                return None
        return self.__point_cloud
