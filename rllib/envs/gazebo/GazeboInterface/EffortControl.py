#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '20/11/2017'
__copyright__ = "RR"
__all__ = [
    'EffortControl',
]

import os
import sys
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from rr_robot_plugin.srv import *

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))


class EffortControl(object):
    def __init__(self, effCmdTopic=None, posStateTopic=None, velStateTopic=None, effStateTopic=None):
        object.__init__(self)
        assert isinstance(effCmdTopic, str) or effCmdTopic is None
        assert isinstance(posStateTopic, str) or posStateTopic is None
        assert isinstance(velStateTopic, str) or velStateTopic is None
        assert isinstance(effStateTopic, str) or effStateTopic is None
        if rospy.get_node_uri() is None:
            rospy.init_node("GazeboInterface",
                            log_level=rospy.INFO,
                            argv=rospy.myargv(argv=sys.argv),
                            anonymous=True)
            rospy.loginfo("EffortControl: Ros node initial.")
        if effCmdTopic:
            self.__pub = rospy.Publisher(effCmdTopic, Float64MultiArray, queue_size=10)
            rospy.loginfo("EffortControl: Initial position control topic '%s'." % (effCmdTopic,))
        if effStateTopic:
            self.__eff = np.array([])
            self.__effSub = rospy.Subscriber(effStateTopic, Float64MultiArray, self.__effStateCallback)
            rospy.loginfo("EffortControl: Initial effort state topic '%s'." % (effStateTopic,))
        if velStateTopic:
            self.__vel = np.array([])
            self.__velSub = rospy.Subscriber(velStateTopic, Float64MultiArray, self.__velStateCallback)
            rospy.loginfo("EffortControl: Initial velocity state topic '%s'." % (velStateTopic,))
        if posStateTopic:
            self.__pos = np.array([])
            self.__posSub = rospy.Subscriber(posStateTopic, Float64MultiArray, self.__posStateCallback)
            rospy.loginfo("EffortControl: Initial position state topic '%s'." % (posStateTopic,))

    def __effStateCallback(self, eff):
        self.__eff = eff.data

    def __velStateCallback(self, vel):
        self.__vel = np.rad2deg(vel.data)

    def __posStateCallback(self, pos):
        self.__pos = np.rad2deg(pos.data)

    def pubEffort(self, eff):
        assert isinstance(eff, (np.ndarray, list, tuple))
        assert len(eff) == 6
        data = Float64MultiArray()
        data.data = eff
        self.__pub.publish(data)

    @property
    def currentEff(self):
        return np.array(self.__eff, np.float64)

    @property
    def currentVel(self):
        return np.array(self.__vel, np.float64)

    @property
    def currentPos(self):
        return np.array(self.__pos, np.float64)

