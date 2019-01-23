#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '31/10/2017'
__copyright__ = "RR"
__all__ = [
    'RobotControl',
]

import os
import time
import sys
import numpy as np
import rospy
from rr_robot_plugin.msg import *
from rr_robot_plugin.srv import *

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))


class RobotControl(object):
    def __init__(self, robot_name, dof):
        object.__init__(self)
        self.__robot_name = robot_name
        self.__dof = dof
        # ros init
        if rospy.get_node_uri() is None:
            rospy.init_node("GazeboInterface",
                            log_level=rospy.INFO,
                            argv=rospy.myargv(argv=sys.argv),
                            anonymous=True)
            rospy.loginfo("RobotControl: Ros node init.")

    @property
    def isMoving(self):
        """
        Returns if robot is moving
        -------

        """
        rospy.wait_for_service(
            '/' + str(self.__robot_name) + '/get_is_moving_state')
        try:
            req = rospy.ServiceProxy(
                '/' + str(self.__robot_name) + '/get_is_moving_state', GetBool)
            res = req()
            success = res.success
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def setStop(self, wait=True):
        """

        Returns: if command received success
        -------

        """
        rospy.wait_for_service(
            '/' + str(self.__robot_name) + '/set_stop_command')
        try:
            req = rospy.ServiceProxy(
                '/' + str(self.__robot_name) + '/set_stop_command', SetBool)
            res = req()
            success = res.success
            if success:
                if wait:
                    while self.isMoving:
                        time.sleep(0.1)
            else:
                rospy.logwarn("RobotControl: setStop: %r fail." )
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def setSpeedRate(self, rate):
        """

        Parameters
        ----------
        rate: range is 0 to 100, represent max vel * % rate

        Returns if command received success
        -------

        """
        rospy.wait_for_service(
            '/' + str(self.__robot_name) + '/set_speed_rate_command')
        try:
            req = rospy.ServiceProxy(
                '/' + str(self.__robot_name) + '/set_speed_rate_command', SetFloat64)
            res = req(rate)
            success = res.success
            if not success:
                rospy.logwarn("RobotControl: setSpeedRate: %r fail." % (rate,))
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def setJointPos(self, joint_pos, wait=False):
        """

        Parameters
        ----------
        joint_pos: [j1_deg, j2_deg, j3_deg, j4_deg, ...]
        wait: if return function until robot move done

        Returns if command received success
        -------

        """
        assert isinstance(joint_pos, (np.ndarray, list, tuple))
        assert len(joint_pos) == self.__dof

        rospy.wait_for_service(
            '/' + str(self.__robot_name) + '/set_joint_pos_command')
        try:
            req = rospy.ServiceProxy(
                '/' + str(self.__robot_name) + '/set_joint_pos_command', SetJointsPath)
            data = JointsPath()
            joints = Joints()
            joints.data = np.deg2rad(joint_pos)
            data.jntsPath.append(joints)
            res = req(data)
            success = res.success
            if success:
                if wait:
                    while self.isMoving:
                        time.sleep(0.1)
            else:
                rospy.logwarn("RobotControl: setJointAngle: %r fail." % (joint_pos,))
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def setJointPath(self, joint_path, wait=False):
        """

        Parameters
        ----------
        joint_path: [[j1_deg, j2_deg, j3_deg, j4_deg, ...], [j1_deg, j2_deg, j3_deg, j4_deg, ...], ...]
        wait: if wait robot move done return function

        Returns if send command success
        -------

        """
        assert isinstance(joint_path, (np.ndarray, list, tuple))
        path = np.array(joint_path, dtype=np.float64)
        assert path.shape[0] > 0 and path.shape[1] == self.__dof
        rospy.wait_for_service(
            '/' + str(self.__robot_name) + '/set_joint_pos_command')
        try:
            req = rospy.ServiceProxy(
                '/' + str(self.__robot_name) + '/set_joint_pos_command', SetJointsPath)
            data = JointsPath()
            for index in xrange(len(path)):
                joints = Joints()
                joints.data = np.deg2rad(path[index])
                data.jntsPath.append(joints)
            res = req(data)
            success = res.success
            if success:
                if wait:
                    while self.isMoving:
                        time.sleep(0.1)
            else:
                rospy.logwarn("RobotControl: setJointPath: %r fail." % (joint_path,))
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def setRobotPos(self, robot_pos, wait=False):
        """

        Parameters
        ----------
        robot_pos: [x_mm, y_mm, z_mm, u_deg, v_deg, w_deg]
        wait: if wait robot move done return function

        Returns if success ik solution
        -------

        """
        assert isinstance(robot_pos, (np.ndarray, list, tuple))
        assert len(robot_pos) == 6
        rospy.wait_for_service(
            '/' + str(self.__robot_name) + '/set_robot_pos_command')
        try:
            req = rospy.ServiceProxy(
                '/' + str(self.__robot_name) + '/set_robot_pos_command', SetCoordinatePath)
            data = CoordinatePath()
            coord = Coordinate()
            coord.data[:3] = np.dot(robot_pos[:3], 0.001)
            coord.data[3:] = np.deg2rad(robot_pos[3:])
            data.crdPath.append(coord)
            res = req(data)
            success = res.success
            if success:
                if wait:
                    while self.isMoving:
                        time.sleep(0.1)
            else:
                rospy.logwarn("RobotControl: setRobotPos: %r fail." % (robot_pos,))
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def setRobotPath(self, robot_path, wait=False):
        """

        Parameters
        ----------
        robot_path: [[x_mm, y_mm, z_mm, u_deg, v_deg, w_deg],
                    [x_mm, y_mm, z_mm, u_deg, v_deg, w_deg], ...]
        wait: if wait robot move done return function

        Returns if all success ik solution
        -------

        """
        assert isinstance(robot_path, (np.ndarray, list, tuple))
        path = np.array(robot_path, dtype=np.float64)
        assert path.shape[0] > 0 and path.shape[1] == 6
        rospy.wait_for_service(
            '/' + str(self.__robot_name) + '/set_robot_pos_command')
        try:
            req = rospy.ServiceProxy(
                '/' + str(self.__robot_name) + '/set_robot_pos_command', SetCoordinatePath)
            data = CoordinatePath()
            for index in xrange(len(path)):
                coord = Coordinate()
                coord.data[:3] = np.dot(path[index][:3], 0.001)
                coord.data[3:] = np.deg2rad(path[index][3:])
                data.crdPath.append(coord)
            res = req(data)
            success = res.success
            if success:
                if wait:
                    while self.isMoving:
                        time.sleep(0.1)
            else:
                rospy.logwarn("RobotControl: setRobotPath: %r fail." % (robot_path,))
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def setLinePath(self, robot_path, end_rot, wait=True, eq_path=100, is_obtuse=False):
        """

        Parameters
        ----------
        robot_path: [[x1_mm, y1_mm, z1_mm], [x2_mm, y2_mm, z2_mm], [x3_mm, y3_mm, z3_mm], ...]
        end_rot: [u_deg, v_deg, w_deg]
        wait: if wait robot move done return function
        eq_path: eplace start point to via point distance(mm)
        is_obtuse: if rotate in obtuse

        Returns: if command received success
        -------

        """
        assert isinstance(robot_path, (np.ndarray, list, tuple))
        path = np.array(robot_path, dtype=np.float64)
        assert path.shape[0] > 0 and path.shape[1] == 3
        rospy.wait_for_service(
            '/' + str(self.__robot_name) + '/set_line_path_command')
        try:
            req = rospy.ServiceProxy(
                '/' + str(self.__robot_name) + '/set_line_path_command', SetLinePath)
            vectors = []
            for index in xrange(len(path)):
                vector = Vector()
                vector.x = path[index][0] * 0.001
                vector.y = path[index][1] * 0.001
                vector.z = path[index][2] * 0.001
                vectors.append(vector)
            rotation = Rotation(np.deg2rad(end_rot[2]), np.deg2rad(end_rot[1]), np.deg2rad(end_rot[0]))
            res = req(eq_path * 0.001, vectors, rotation, is_obtuse)
            success = res.success
            if success:
                if wait:
                    while self.isMoving:
                        time.sleep(0.1)
            else:
                rospy.logwarn("RobotControl: setLinePath: %r fail." % (robot_path,))
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def setArcPath(self, point1, point2, end_rot, is_circle=False, wait=True, is_obtuse=False):
        """

        Parameters
        ----------
        point1: [x1_mm, y1_mm, z1_mm]
        point2: [x2_mm, y2_mm, z2_mm]
        end_rot: [u_deg, v_deg, w_deg]
        is_circle: if run circle
        wait: if wait robot move done return function
        is_obtuse: if rotate in obtuse

        Returns: if command received success
        -------

        """
        assert isinstance(point1, (np.ndarray, list, tuple))
        assert isinstance(point2, (np.ndarray, list, tuple))
        rospy.wait_for_service(
            '/' + str(self.__robot_name) + '/set_arc_path_command')
        try:
            req = rospy.ServiceProxy(
                '/' + str(self.__robot_name) + '/set_arc_path_command', SetArcPath)
            vector1 = Vector()
            vector1.x = point1[0] * 0.001
            vector1.y = point1[1] * 0.001
            vector1.z = point1[2] * 0.001
            vector2 = Vector()
            vector2.x = point2[0] * 0.001
            vector2.y = point2[1] * 0.001
            vector2.z = point2[2] * 0.001
            rotation = Rotation(np.deg2rad(end_rot[2]), np.deg2rad(end_rot[1]), np.deg2rad(end_rot[0]))
            res = req(is_circle, vector1, vector2, rotation, is_obtuse)
            success = res.success
            if success:
                if wait:
                    while self.isMoving:
                        time.sleep(0.1)
            else:
                rospy.logwarn("RobotControl: setArcPath: %r fail." % (point1,))
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def getJointPos(self):
        """

        Returns np.array([j1_deg, j2_deg, j3_deg, ...])
        -------

        """
        rospy.wait_for_service(
            '/' + str(self.__robot_name) + '/get_joint_pos_state')
        try:
            req = rospy.ServiceProxy(
                '/' + str(self.__robot_name) + '/get_joint_pos_state', GetJointsPos)
            res = req()
            return np.array(np.rad2deg(res.jntPos.data[:]), dtype=np.float64)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def getRobotPos(self):
        """

        :return: np.array([x_mm, y_mm, z_mm, u_deg, v_deg, w_deg])
        """
        rospy.wait_for_service(
            '/' + str(self.__robot_name) + '/get_robot_pos_state')
        try:
            req = rospy.ServiceProxy(
                '/' + str(self.__robot_name) + '/get_robot_pos_state', GetCoordinate)
            res = req()
            robot_pos = np.array(res.crdPos.data[:], dtype=np.float64)
            robot_pos[:3] = np.dot(robot_pos[:3], 1000)
            robot_pos[3:] = np.rad2deg(robot_pos[3:])
            return robot_pos
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

if __name__ == "__main__":
    robot = RobotControl('kent6_v3', 6)
    speed_rate = 80
    robot.setSpeedRate(speed_rate)
    robot.setJointPos([-90, -30, 30, 0, 0, 0], wait=True)
    jntsPath = [[-70, 30, -30, 0, 0, 0],
                [-50, -30, 30, 0, 0, 0],
                [-30, 30, -30, 0, 0, 0],
                [-10, -30, 30, 0, 0, 0],
                [10, 30, -30, 0, 0, 0],
                [30, -30, 30, 0, 0, 0],
                [50, 30, -30, 0, 0, 0],
                [70, -30, 30, 0, 0, 0],
                [90, 30, -30, 0, 0, 0]
                ]
    crdPath = [
               [187.0, -515.0, 595.0, 20.0, -90.0, 90.0],
               [185.0, -220.0, 595.0, 40.0, -90.0, 90.0],
               [475.0, -274.0, 595.0, -120.0, -90.0, -90.0],
               [284.0, -50.0, 595.0, -100.0, -90.0, -90.0],
               [540.0, 95.0, 595.0, -80.0, -90.0, -90.0],
               [250, 144.0, 595.0, -60.0, -90.0, -90.0],
               [352.0, 420.0, 595.0, -40.0, -90.0, -90.0],
               [99.0, 271.0, 595.0, 160.0, -90.0, 90.0],
               [0.0, 548.0, 595.0, 0.0, -90.0, -90.0]
               ]
    while True:
        robot.setJointPath(jntsPath, wait=True)
        for i in xrange(len(jntsPath)):
            robot.setJointPos(jntsPath[i], wait=True)
        robot.setRobotPath(crdPath, wait=True)
        for i in xrange(len(crdPath)):
            robot.setRobotPos(crdPath[i], wait=True)

        robot.setJointPos([-90, 0, 0, 0, 0, 0], wait=True)
        print("joint: ", robot.getJointPos())
        robot.setJointPos([90, 0, 0, 0, 0, 0], wait=False)
        time.sleep(0.3)
        robot.setJointPos([-90, 0, 0, 0, 0, 0], wait=True)
        robot.setRobotPos([550, 0, 300, 0, -90, 180], wait=True)
        print("coord: ", robot.getRobotPos())
        robot.setLinePath([[550, 0, 600]], [0, -90, 180], wait=False)
        time.sleep(0.4)
        robot.setStop()
        time.sleep(0.5)
        robot.setRobotPos([500, 0, 500, 0, -90, 180], wait=True)
        robot.setArcPath([500, 150, 300], [500, 0, 300], [0, -90, 180], is_circle=True, wait=True)
        time.sleep(0.5)