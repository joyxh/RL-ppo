#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'zdy'
__version__ = '1.0.0'
__date__ = '20/11/2017'
__copyright__ = "RR"
__all__ = [
    'WorldControl',
]

import os
import sys
import numpy as np
import rospy
from rr_robot_plugin.srv import *
from geometry_msgs.msg import Twist
from std_msgs.msg import *
#from ModelEdit import *

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))


class WorldControl(object):
    def __init__(self):
        object.__init__(self)
        # ros init
        if rospy.get_node_uri() is None:
            rospy.init_node("GazeboInterface",
                            log_level=rospy.INFO,
                            argv=rospy.myargv(argv=sys.argv),
                            anonymous=True)
            rospy.loginfo("WorldControl: Ros node init.")
        self.__server_name = "gzserver"

    def addBoxModel(self, modelName, initialPos, mass=1, x=1, y=1, z=1, color='Grey'):
        """

        :param modelName: name you want name to model
        :param initialPos: [x_mm, y_mm, z_mm, u_deg, v_deg, w_deg]
        :param mass: mass of model
        :param x:
        :param y:
        :param z:
        :param color: choose in ~/.gazebo/models/materials/scripts/Pi.material
        :return: if add success
        """
        xml = createBoxModel(mass, x, y, z, color)
        success = self.spawnSdfString(modelName, initialPos, xml)
        if not success:
            rospy.logerr("WorldControl: addBoxModel: %r fail." % (modelName,))
        return success

    def addBoxVisual(self, modelName, initialPos, x=1, y=1, z=1, color='Grey'):
        """

        :param modelName: name you want name to model
        :param initialPos: [x_mm, y_mm, z_mm, u_deg, v_deg, w_deg]
        :param x:
        :param y:
        :param z:
        :param color: choose in ~/.gazebo/models/materials/scripts/Pi.material
        :return: if add success
        """
        xml = createBoxVisual(x, y, z, color)
        success = self.spawnSdfString(modelName, initialPos, xml)
        if not success:
            rospy.logerr("WorldControl: addBoxVisual: %r fail." % (modelName,))
        return success

    def addCylinderModel(self, modelName, initialPos, mass=1, radius=1, length=1, color='Grey'):
        """

        :param modelName: name you want name to model
        :param initialPos: : [x_mm, y_mm, z_mm, u_deg, v_deg, w_deg]
        :param mass: mass of model
        :param radius: roundness radius
        :param length:
        :param color: choose in ~/.gazebo/models/materials/scripts/Pi.material
        :return: if add success
        """
        xml = createCylinderModel(mass, radius, length, color)
        success = self.spawnSdfString(modelName, initialPos, xml)
        if not success:
            rospy.logerr("WorldControl: addCylinderModel: %r fail." % (modelName,))
        return success

    def addCylinderVisual(self, modelName, initialPos, radius=1, length=1, color='Grey'):
        """

        :param modelName: name you want name to model
        :param initialPos: position which model add in
        :param radius:
        :param length:
        :param color: choose in ~/.gazebo/models/materials/scripts/Pi.material
        :return: if add success
        """
        xml = createCylinderVisual(radius, length, color)
        success = self.spawnSdfString(modelName, initialPos, xml)
        if not success:
            rospy.logerr("WorldControl: addCylinderVisual: %r fail." % (modelName,))
        return success

    def addSphereModel(self, modelName, initialPos, mass=1, radius=1, color='Grey'):
        """

        :param modelName: name you want name to model
        :param initialPos: position which model add in
        :param mass: mass of model
        :param radius: roundness radius
        :param color: choose in ~/.gazebo/models/materials/scripts/Pi.material
        :return: if add success
        """
        xml = createSphereModel(mass, radius, color)
        success = self.spawnSdfString(modelName, initialPos, xml)
        if not success:
            rospy.logerr("WorldControl: addSphereModel: %r fail." % (modelName,))
        return success

    def addSphereVisual(self, modelName, initialPos, radius=1, color='Grey'):
        """

        :param modelName: name you want name to model
        :param initialPos: position which model add in
        :param radius:
        :param color: choose in ~/.gazebo/models/materials/scripts/Pi.material
        :return: if add success
        """
        xml = createSphereVisual(radius, color)
        success = self.spawnSdfString(modelName, initialPos, xml)
        if not success:
            rospy.logerr("WorldControl: addSphereVisual: %r fail." % (modelName,))
        return success

    def deleteModel(self, modelName):
        """

        :param modelName: model name in gazebo
        :return: if delete success
        """
        assert isinstance(modelName, str)
        rospy.wait_for_service("/" + self.__server_name + "/delete_model")
        try:
            req = rospy.ServiceProxy("/" + self.__server_name + "/delete_model", DelModel)
            res = req(modelName)
            success = res.success
            if not success:
                rospy.logerr("WorldControl: deleteModel: %r fail." % (modelName,))
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def getExistModels(self):
        """

        :return: list exist models names in gazebo
        """
        rospy.wait_for_service("/" + self.__server_name + "/get_exist_models")
        try:
            req = rospy.ServiceProxy("/" + self.__server_name + "/get_exist_models", GetExistModels)
            res = req()
            return res.modelName
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def getLinkPos(self, linkName):
        rospy.wait_for_service("/" + self.__server_name + "/get_link_pos")
        try:
            req = rospy.ServiceProxy("/" + self.__server_name + "/get_link_pos", GetLinkPos)
            res = req(linkName)
            if res.success:
                return np.array([res.pos[0] * 1000.0, res.pos[1] * 1000.0, res.pos[2] * 1000.0,
                                 np.rad2deg(res.pos[5]), np.rad2deg(res.pos[4]), np.rad2deg(res.pos[3])])
            else:
                rospy.logerr("WorldControl: getLinkPos: %r fail." % (linkName,))
                # TODO: if no success how to do
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def getModelPos(self, modelName):
        """

        :param modelName: model name in gazebo
        :return: model pos in gazebo world
        """
        rospy.wait_for_service("/" + self.__server_name + "/get_model_pos")
        try:
            req = rospy.ServiceProxy("/" + self.__server_name + "/get_model_pos", GetModelPos)
            res = req(modelName)
            if res.success:
                return np.array([res.pos[0] * 1000.0, res.pos[1] * 1000.0, res.pos[2] * 1000.0,
                                 np.rad2deg(res.pos[5]), np.rad2deg(res.pos[4]), np.rad2deg(res.pos[3])])
            else:
                rospy.logerr("WorldControl: getModelPos: %r fail." % (modelName,))
            # TODO: if no success how to do
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def getModelVel(self, modelName):
        """

        :param modelName:
        :return: velocity(x_mm_dot, y_mm_dot, z_mm_dot, u_deg_dot, v_deg_dot, w_deg_dot)
        """
        rospy.wait_for_service("/" + self.__server_name + "/get_model_vel")
        try:
            req = rospy.ServiceProxy("/" + self.__server_name + "/get_model_vel", GetModelVel)
            res = req(modelName)
            if res.success:
                return np.array([res.vel.linear.x * 1000.0,
                                 res.vel.linear.y * 1000.0,
                                 res.vel.linear.z * 1000.0,
                                 np.rad2deg(res.vel.angular.z),
                                 np.rad2deg(res.vel.angular.y),
                                 np.rad2deg(res.vel.angular.x)])
            else:
                rospy.logerr("WorldControl: getModelVel: %r fail." % (modelName,))
            # TODO: if no success how to do
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def setModelGravity(self, modelName, isGravity):
        """

        :param modelName: model name in gazebo
        :param isGravity: is model affect by gravity
        :return:
        """
        assert isinstance(modelName, str)
        assert isinstance(isGravity, bool)
        rospy.wait_for_service("/" + self.__server_name + "/set_model_gravity")
        try:
            req = rospy.ServiceProxy("/" + self.__server_name + "/set_model_gravity", SetModelBool)
            res = req(modelName, isGravity)
            success = res.success
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def setModelStatic(self, modelName, isStatic):
        """

        :param modelName: model name in gazebo
        :param isStatic: if model static
        :return:
        """
        assert isinstance(modelName, str)
        assert isinstance(isStatic, bool)
        rospy.wait_for_service("/" + self.__server_name + "/set_model_static")
        try:
            req = rospy.ServiceProxy("/" + self.__server_name + "/set_model_static", SetModelBool)
            res = req(modelName, isStatic)
            success = res.success
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def setLinkPos(self, linkName, pos):
        assert isinstance(linkName, str)
        assert isinstance(pos, (np.ndarray, list, tuple))
        rospy.wait_for_service("/" + self.__server_name + "/set_link_pos")
        try:
            req = rospy.ServiceProxy("/" + self.__server_name + "/set_link_pos", SetLinkPos)
            pos = self._poseConvert(pos)
            res = req(linkName, pos)
            success = res.success
            if not success:
                rospy.logerr("WorldControl: setLinkPos: %r fail." % (linkName,))
            return success
            # TODO: if no success how to do
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def setModelPos(self, modelName, pos):
        """

        :param modelName: model name in gazebo
        :param pos: set position in gazebo reference frame is world
        :return: if setting success
        """
        assert isinstance(modelName, str)
        assert isinstance(pos, (np.ndarray, list, tuple))
        rospy.wait_for_service("/" + self.__server_name + "/set_model_pos")
        try:
            req = rospy.ServiceProxy("/" + self.__server_name + "/set_model_pos", SetModelPos)
            pos = self._poseConvert(pos)
            res = req(modelName, pos)
            success = res.success
            if not success:
                rospy.logerr("WorldControl: setModelPos: %r fail." % (modelName,))
            return success
            # TODO: if no success how to do
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def setModelVel(self, modelName, vel):
        """

        :param modelName: model name in gazebo
        :param vel: [x_mm_dot, y_mm_dot, z_mm_dot, u_deg_dot, v_deg_dot, w_deg_dot]
        :return: if setting success
        """
        assert isinstance(modelName, str)
        assert isinstance(vel, (np.ndarray, list, tuple))
        assert len(vel) == 6
        rospy.wait_for_service("/" + self.__server_name + "/set_model_pos")
        try:
            req = rospy.ServiceProxy("/" + self.__server_name + "/set_model_pos", SetModelPos)
            twist = Twist()
            twist.linear.x = vel[0] * 0.001
            twist.linear.y = vel[1] * 0.001
            twist.linear.z = vel[2] * 0.001
            twist.angular.z = np.deg2rad(vel[5])
            twist.angular.y = np.deg2rad(vel[4])
            twist.angular.x = np.deg2rad(vel[3])
            res = req(modelName, twist)
            if res.success:
                return np.array([res.pos[0] * 1000.0, res.pos[1] * 1000.0, res.pos[2] * 1000.0,
                                 np.rad2deg(res.pos[5]), np.rad2deg(res.pos[4]), np.rad2deg(res.pos[3])])
            else:
                rospy.logerr("WorldControl: setModelVel: %r fail." % (modelName,))
                # TODO: if no success how to do
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def spawnSdfFile(self, modelName, initialPos, filePath):
        """

        :param modelName: rename the model
        :param initialPos: pos in gazebo reference frame is world
        :param filePath:
        :return: if setting success
        """
        assert isinstance(modelName, str)
        assert isinstance(filePath, str)
        assert isinstance(initialPos, (np.ndarray, list, tuple))
        rospy.wait_for_service("/" + self.__server_name + "/spawn_sdf_file")
        try:
            req = rospy.ServiceProxy("/" + self.__server_name + "/spawn_sdf_file", SpawnModel)
            initialPos = self._poseConvert(initialPos)
            res = req(modelName, filePath, initialPos)
            success = res.success
            if not success:
                rospy.logerr("WorldControl: spawnSdfFile: %r fail." % (modelName,))
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def spawnUrdfFile(self, modelName, initialPos, filePath):
        """

        :param modelName: rename the model
        :param initialPos: pos in gazebo reference frame is world
        :param filePath:
        :return: if setting success
        """
        assert isinstance(modelName, str)
        assert isinstance(filePath, str)
        assert isinstance(initialPos, (np.ndarray, list, tuple))
        rospy.wait_for_service("/" + self.__server_name + "/spawn_urdf_file")
        try:
            req = rospy.ServiceProxy("/" + self.__server_name + "/spawn_urdf_file", SpawnModel)
            initialPos = self._poseConvert(initialPos)
            res = req(modelName, filePath, initialPos)
            success = res.success
            if not success:
                rospy.logerr("WorldControl: spawnSdfFile: %r fail." % (modelName,))
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def spawnSdfString(self, modelName, initialPos, string):
        """

        :param modelName: rename the model
        :param initialPos: pos in gazebo reference frame is world
        :param string:
        :return: if setting success
        """
        assert isinstance(modelName, str)
        assert isinstance(string, str)
        assert isinstance(initialPos, (np.ndarray, list, tuple))
        rospy.wait_for_service("/" + self.__server_name + "/spawn_sdf_string")
        try:
            req = rospy.ServiceProxy("/" + self.__server_name + "/spawn_sdf_string", SpawnModel)
            initialPos = self._poseConvert(initialPos)
            res = req(modelName, string, initialPos)
            success = res.success
            if not success:
                rospy.logerr("WorldControl: spawnSdfFile: %r fail." % (modelName,))
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def spawnUrdfString(self, modelName, initialPos, string):
        """

        :param modelName: rename the model
        :param initialPos: pos in gazebo reference frame is world
        :param string:
        :return: if setting success
        """
        assert isinstance(modelName, str)
        assert isinstance(string, str)
        assert isinstance(initialPos, (np.ndarray, list, tuple))
        rospy.wait_for_service("/" + self.__server_name + "/spawn_urdf_string")
        try:
            req = rospy.ServiceProxy("/" + self.__server_name + "/spawn_urdf_string", SpawnModel)
            initialPos = self._poseConvert(initialPos)
            res = req(modelName, string, initialPos)
            success = res.success
            if not success:
                rospy.logerr("WorldControl: spawnSdfFile: %r fail." % (modelName,))
            return success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def _poseConvert(self, robotPose):
        """
        :function: convert robot pose uvw to gazebo pose rpy
        :param robotPose: pose_1x6 or pose_6x1 in robot frame like [x_mm, y_mm, z_mm, u_deg, v_deg, w_deg]
        :return : pose_1x6 in gazebo frame like [x_m, y_m, z_m, u_rad, v_rad, w_rad]
        """
        tempPose = np.zeros(6, dtype=np.float32)
        robotPose = np.array(robotPose, dtype=np.float32).reshape(-1)
        assert len(robotPose) == 6
        tempPose[0:3] = robotPose[0:3]
        tempPose[3] = robotPose[5]
        tempPose[4] = robotPose[4]
        tempPose[5] = robotPose[3]
        tempPose[0:3] = np.dot(tempPose[0:3], 0.001)
        tempPose[3:6] = np.deg2rad(tempPose[3:6])
        return tempPose
    
if __name__ == '__main__':
    my_world = WorldControl()
    my_world.spawnSdfFile("Box", [1.0, 0, 0, 0, 0, 0], "/home/pi/.gazebo/models/box/model.sdf")
