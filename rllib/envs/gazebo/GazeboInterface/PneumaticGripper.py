#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '20/11/2017'
__copyright__ = "RR"
__all__ = [
    'PneumaticGripper'
]

import os
import sys

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))
import rospy
from rr_robot_plugin.srv import *


class PneumaticGripper(object):
    def __init__(self, controlServiceName, stateServiceName):
        object.__init__(self)
        assert isinstance(controlServiceName, str)
        assert isinstance(stateServiceName, str)
        self.__control_service_name = controlServiceName
        self.__state_service_name = stateServiceName
        if rospy.get_node_uri() is None:
            rospy.init_node("GazeboInterface",
                            log_level=rospy.INFO,
                            argv=rospy.myargv(argv=sys.argv),
                            anonymous=True)
        rospy.loginfo("PneumaticGripper: Ros node initial.")

    def catch(self):
        rospy.wait_for_service(self.__control_service_name)
        try:
            req = rospy.ServiceProxy(self.__control_service_name, SetBool)
            res = req(True)
            success = res.success
            if not success:
                rospy.logerr("PneumaticGripper: catch fail.")
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def release(self):
        rospy.wait_for_service(self.__control_service_name)
        try:
            req = rospy.ServiceProxy(self.__control_service_name, SetBool)
            res = req(False)
            success = res.success
            if not success:
                rospy.logerr("PneumaticGripper: release fail.")
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def isGripperCatch(self):
        rospy.wait_for_service(self.__state_service_name)
        try:
            req = rospy.ServiceProxy(self.__state_service_name, GetBool)
            res = req()
            state = res.success
            return state
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
