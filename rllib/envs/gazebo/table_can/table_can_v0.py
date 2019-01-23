import math
from gym import utils, spaces
from gym.utils import seeding
import numpy as np

# from gym_gazebo_rr.envs.utils.GazeboInterface import Camera, RobotControl, PneumaticGripper
# from gym_gazebo_rr.envs.utils import GazeboInterface
from envs.gazebo import gazebo_env
from envs.gazebo.GazeboInterface.Camera import Camera
from envs.gazebo.GazeboInterface.RobotControl import RobotControl
from envs.gazebo.GazeboInterface.PneumaticGripper import PneumaticGripper
from envs.gazebo.GazeboInterface.WorldControl import WorldControl

ROBOT_NAME = 'rr_arm'
DOF = 6
ROBOT_INIT_POSE = [-200, -300, 500, -90, 0, 180]
CAN_INIT_POSE = [0, -450, 775, 0, 0, 0]
# CAN_LINK = 'can::circle::circle_link'
CAN_LINK = 'can::link'
CAN_D = 100 # diameter of can
CAN_H = 100 # height of can
ROBOT_BASE_LINK = 'rr_arm::rr6::table_link'

class TableCanv0Env(gazebo_env.GazeboEnv):
    def __init__(self):
        
        # Launch the simulation with the given launchfile name
        launchfile_path = 'table_can_env.launch'
        gazebo_env.GazeboEnv.__init__(self, launchfile_path)
       
        # setup robot
        self._robot = RobotControl(robot_name=ROBOT_NAME, dof=DOF)
        self._gripper = PneumaticGripper("/gripper/pneumatic_gripper/box_link/pneumatic_gripper_control",
                               "/gripper/pneumatic_gripper/box_link/pneumatic_gripper_state")
        
        self._cam1 = Camera(rgbImageTopic="/cam_1/camera/link/rgb/image")
        self._cam2 = Camera(rgbImageTopic="/cam_2/camera/link/rgb/image")
        self._world = WorldControl()
        self.robot_base = self._world.getLinkPos(linkName=ROBOT_BASE_LINK)
        self.can_pos = self._world.getLinkPos(linkName=CAN_LINK)
        # Seed the environment
        self.action_space = spaces.Discrete(2) #x,y,z,u,v,w
        self.reward_range = (-np.inf, np.inf)
        self.observation = None
        self.reward = 0
        self.done = None
        self.info = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, reward_type):
        # current_pos = self._robot.getRobotPos()
        # self._robot.setRobotPos(current_pos + action, wait=True)
        # if self._action_judge(action):
        #     self._robot.setRobotPos(action, wait=True)
        # else:
        #     return
        self._robot.setRobotPos(action, wait=True)
        img1 = self._cam1.getRGBImage()
        img2 = self._cam2.getRGBImage()
        robot_angle = self._robot.getJointPos()
        self.robot_pos = self._robot.getRobotPos() + self.robot_base
        self.observation = [img1, img2, robot_angle, self.robot_pos]

        self.can_pos = self._world.getLinkPos(linkName=CAN_LINK)
        self.info = self.can_pos

        if self._robot_near_can():
            self.done = True
        else:
            self.done = False

        if 1 == reward_type: 
            self._reword_type1()
        elif 2 == reward_type:
            self._reword_type2()
        else:
            self._reword_type3()

        return self.observation, self.reward, self.done, self.info

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        self._robot.setRobotPos(ROBOT_INIT_POSE, wait=True)
        return True

    def _action_judge(self, action):
        # Determine if the next position of the robot is legal
        pass

    def _robot_near_can(self):
        allowed_dist = math.sqrt((CAN_H/2)**2 + CAN_D**2)
        x_bias = self.robot_pos[0] - self.can_pos[0]
        y_bias = self.robot_pos[1] - self.can_pos[1]
        z_bias = self.robot_pos[2] - self.can_pos[2]
        self.distance = math.sqrt(x_bias**2 + y_bias**2 + z_bias**2)
        if self.distance < allowed_dist:
            return True
        else:
            return False

    def _reword_type1(self):
        if self.done:
            self.reward = 1
        else:
            self.reward = 0
    
    def _reword_type2(self):
        if self.done:
            self.reward = 0
        else:
            self.reward -= 1

    def _reword_type3(self):
        self.reward = -self.distance
