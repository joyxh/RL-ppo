#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '20/11/2017'
__copyright__ = "RR"
__all__ = [
    'PositionControl',
]

import os
import sys
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from rr_robot_plugin.srv import *

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))


class PositionControl(object):
    def __init__(self, posCmdTopic=None, posStateTopic=None, velStateTopic=None):
        object.__init__(self)
        assert isinstance(posCmdTopic, str) or posCmdTopic is None
        assert isinstance(posStateTopic, str) or posStateTopic is None
        assert isinstance(velStateTopic, str) or velStateTopic is None
        if rospy.get_node_uri() is None:
            rospy.init_node("GazeboInterface",
                            log_level=rospy.INFO,
                            argv=rospy.myargv(argv=sys.argv),
                            anonymous=True)
            rospy.loginfo("PositionControl: Ros node initial.")
        if posCmdTopic:
            self.__pub = rospy.Publisher(posCmdTopic, Float64MultiArray, queue_size=10)
            rospy.loginfo("PositionControl: Initial position control topic '%s'." % (posCmdTopic,))
        if velStateTopic:
            self.__vel = np.array([])
            self.__sub = rospy.Subscriber(velStateTopic, Float64MultiArray, self.__velStateCallback)
            rospy.loginfo("PositionControl: Initial velocity state topic '%s'." % (velStateTopic,))
        if posStateTopic:
            self.__pos = np.array([])
            self.__sub = rospy.Subscriber(posStateTopic, Float64MultiArray, self.__posStateCallback)
            rospy.loginfo("PositionControl: Initial velocity state topic '%s'." % (posStateTopic,))

    def __velStateCallback(self, vel):
        self.__vel = np.rad2deg(vel.data)

    def __posStateCallback(self, pos):
        self.__pos = np.rad2deg(pos.data)

    def pubPosition(self, pos):
        assert isinstance(pos, (np.ndarray, list, tuple))
        assert len(pos) == 6
        data = Float64MultiArray()
        data.data = pos
        self.__pub.publish(data)

    @property
    def currentVel(self):
        return np.array(self.__vel, np.float64)

    @property
    def currentPos(self):
        return np.array(self.__pos, np.float64)

