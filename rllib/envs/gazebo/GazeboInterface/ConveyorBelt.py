#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '20/11/2017'
__copyright__ = "RR"
__all__ = [
    'ConveyorBelt'
]
import os
import sys

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))
import rospy
from rr_robot_plugin.srv import *


class ConveyorBelt(object):
    def __init__(self, modelName):
        """

        :param modelName: name in gazebo
        """
        object.__init__(self)
        assert isinstance(modelName, str)
        self.__model_name = modelName
        if rospy.get_node_uri() is None:
            rospy.init_node("GazeboInterface",
                            log_level=rospy.INFO,
                            argv=rospy.myargv(argv=sys.argv),
                            anonymous=True)
            rospy.loginfo("ConveyorBelt: Ros node initial.")

    def setConveyorVel(self, vel):
        """

        :param vel: conveyor belt velocity
        :return: if setting success
        """
        rospy.wait_for_service("/" + self.__model_name + "/set_conveyor_belt_vel")
        try:
            req = rospy.ServiceProxy("/" + self.__model_name + "/set_conveyor_belt_vel", SetFloat64)
            res = req(vel)
            success = res.success
            if not success:
                rospy.logerr("ConveyorBelt: setConveyorVel fail.")
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def getConveyorVel(self):
        """

        :return: current conveyor belt velocity
        """
        rospy.wait_for_service("/" + self.__model_name + "/get_conveyor_belt_vel")
        try:
            req = rospy.ServiceProxy("/" + self.__model_name + "/get_conveyor_belt_vel", GetFloat64)
            res = req()
            vel = res.data
            return vel
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

