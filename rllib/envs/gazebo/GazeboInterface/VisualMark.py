#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '20/11/2017'
__copyright__ = "RR"
__all__ = [
    'VisualMark',
]

import os
import sys
import rospy
from geometry_msgs.msg import *
from rr_robot_plugin.srv import *

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))


class VisualMark(object):
    def __init__(self):
        object.__init__(self)
        # ros init
        if rospy.get_node_uri() is None:
            rospy.init_node("GazeboInterface",
                            log_level=rospy.INFO,
                            argv=rospy.myargv(argv=sys.argv),
                            anonymous=True)
            rospy.loginfo("VisualMark: Ros node init.")
        self.__color = ['Red', 'Green', 'Blue', 'Purple', 'Red', 'White', 'Yellow']
        self.__client_name = "gzclient"
        for groupName in self.getGroupNames():
            self.delGroup(groupName)

    def addGroup(self, groupName, markType, color):
        """

        :param groupName: name of group
        :param markType: 'POINT' or 'LINE'
        :param color: 'Red', 'Green', 'Blue', 'Purple', 'Red', 'White', 'Yellow'
        :return: if success
        """
        assert isinstance(groupName, str)
        assert markType in ['POINT', 'LINE']
        assert color in self.__color
        rospy.wait_for_service("/" + self.__client_name + "/addGroup")
        try:
            req = rospy.ServiceProxy("/" + self.__client_name + "/addGroup", AddGroup)
            res = req(groupName, markType, "Gazebo/" + color)
            success = res.success
            if not success:
                rospy.logerr("VisualMark: addGroup: %r fail." % (groupName,))
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def delGroup(self, groupName):
        """

        :param groupName: name of group
        :return: if success
        """
        assert isinstance(groupName, str)
        rospy.wait_for_service("/" + self.__client_name + "/delGroup")
        try:
            req = rospy.ServiceProxy("/" + self.__client_name + "/delGroup", DelGroup)
            res = req(groupName)
            success = res.success
            if not success:
                rospy.logerr("VisualMark: delGroup: %r fail." % (groupName,))
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def setPoint(self, groupName, pos):
        """
        reference frame is world
        :param groupName: name of group
        :param pos: [x_mm, y_mm, z_mm]
        :return: if success
        """
        assert isinstance(groupName, str)
        assert len(pos) == 3
        rospy.wait_for_service("/" + self.__client_name + "/setPoint")
        try:
            req = rospy.ServiceProxy("/" + self.__client_name + "/setPoint", SetPoint)
            point = Point()
            point.x = pos[0] * 0.001
            point.y = pos[1] * 0.001
            point.z = pos[2] * 0.001
            res = req(groupName, point)
            success = res.success
            if not success:
                rospy.logerr("VisualMark: setPoint: %r fail." % (groupName,))
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def getGroupNames(self):
        """
        :return: [groupName1, groupName2, ......]
        """
        rospy.wait_for_service("/" + self.__client_name + "/getGroupNames")
        try:
            req = rospy.ServiceProxy("/" + self.__client_name + "/getGroupNames", GetGroupNames)
            res = req()
            return res.data
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)


if __name__ == "__main__":
    vsm = VisualMark()
    raw_input("add line")
    vsm.addGroup("1", "LINE", "Red")
    vsm.setPoint("1", [1000, 1000, 1000])
    vsm.setPoint("1", [1000, 2000, 1000])
    vsm.setPoint("1", [2000, 1000, 1000])
    print(vsm.getGroupNames())
    raw_input("del line")
    for group_name in vsm.getGroupNames():
        vsm.delGroup(group_name)
    print(vsm.getGroupNames())
    raw_input("add point")
    vsm.addGroup("1", "POINT", "Red")
    vsm.setPoint("1", [1000, 1000, 1000])
    vsm.setPoint("1", [1000, 2000, 1000])
    vsm.setPoint("1", [2000, 1000, 1000])
    print(vsm.getGroupNames())
    raw_input("del point")
    for group_name in vsm.getGroupNames():
        vsm.delGroup(group_name)
    print(vsm.getGroupNames())
