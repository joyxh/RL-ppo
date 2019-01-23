import random
import time
import numpy as np

from gym import logger, spaces
from gym.utils import seeding

from envs.gazebo import gazebo_env
from envs.gazebo.GazeboInterface import Camera
from envs.gazebo.GazeboInterface.PneumaticGripper import PneumaticGripper
from envs.gazebo.GazeboInterface.RobotControl import RobotControl
from envs.gazebo.GazeboInterface.WorldControl import WorldControl

ROBOT_NAME = 'rr_arm'
DOF = 6
ROBOT_BASE_LINK = 'rr_arm::rr6::table_link'
GRIPPER_H_MM = 127
OBJECT_H_MM = 25
OBJECT_W_MM = 57.5

IMAGE_SHAPE = (960, 1280, 3)

class PickPlacev0Env(gazebo_env.GazeboEnv):
    def __init__(self, step_type, grab_range_z_mm, random_xy_type):
        """
        Init Gazebo Env and Gazebo Control API
        :param step_type: SetRobot next step , JointAngle or Position
        :param grab_range_z_mm : Robot can grab the object in the grab range(z)
        :param random_xy_type : type of randomly generated (x,y) coordinates
        """
        self.step_type = step_type
        self.grab_range_z_mm = grab_range_z_mm
        self.random_xy_type = random_xy_type
        
        # Launch the simulation with the given launchfile name
        launchfile_path = 'pick_place_env.launch'
        gazebo_env.GazeboEnv.__init__(self, launchfile_path)
       
        # setup robot
        self.robot = RobotControl(robot_name=ROBOT_NAME, dof=DOF)
        self.gripper = PneumaticGripper("/gripper/pneumatic_gripper/box_link/pneumatic_gripper_control",
                               "/gripper/pneumatic_gripper/box_link/pneumatic_gripper_state")
        self._cam_left1 = Camera(rgbImageTopic="/cam_left1/camera/link/rgb/image")
        self._cam_left2 = Camera(rgbImageTopic="/cam_left2/camera/link/rgb/image")
        # self._cam_right1 = Camera(rgbImageTopic="/cam_right1/camera/link/rgb/image")
        # self._cam_right2 = Camera(rgbImageTopic="/cam_right2/camera/link/rgb/image")
        self._world = WorldControl()
        self.robot_base = self._world.getLinkPos(linkName=ROBOT_BASE_LINK)
        self.object_pos = self._world.getModelPos(modelName="cube")
        # Seed the environment
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(DOF,), dtype='float32') #x,y,z,u,v,w
        self.observation_space = spaces.Dict(dict(
            img_left1 = spaces.Box(low=0, high=255, shape=IMAGE_SHAPE, dtype=np.uint8),
            img_left2 = spaces.Box(low=0, high=255, shape=IMAGE_SHAPE, dtype=np.uint8),
            robot_pos = spaces.Box(-np.inf, np.inf, shape=(DOF,), dtype='float32'),
            gripper_pos = spaces.Box(-np.inf, np.inf, shape=(DOF,), dtype='float32'),
        ))
        self.reward_range = (-np.inf, np.inf)

        # set to not displayed in scientific notation
        # np.set_printoptions(suppress=True)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Take a step and return reword....
        Note that objects and robot are in the robot coordinate system.
        :param action: robot pose[x,y,z,u,v,w] or joint angle
        """
        reward = 0
        done = False
        done_reason = 'Not Done'
        # done_type: 0 未完成; 1 正常完成; >1 非正常终止
        done_type = 0
        if self.step_type == 'JA':
            success = self.robot.setJointPos(action, wait=True)
            if not success:
                done = True
                done_reason = 'Set Joint Angle Failed. '
                done_type = 2
        elif self.step_type == 'Pose':
            if self._collision(action):
                done = True
                done_reason = 'May Cause Collision, need to reset .'
                done_type = 3
            else:
                success = self.robot.setRobotPos(action, wait=True)
                if not success:
                    # logger.warn("Set Robot pos failed ! ")
                    done = True
                    done_reason = 'Set Position Failed. '
                    done_type = 2
        else:
            pass

        if self._enter_grab_range():
            reward = 1
            done = True
            done_reason = 'Well Done! Finish the Task! '
            done_type = 1
        observation = self._get_obs()
        info = {'object_pos' : self.object_pos-self.robot_base, 
                'step_type' : self.step_type,
                'grab_range_z_mm' : self.grab_range_z_mm,
                'random_xy_type' : self.random_xy_type,
                'done_reason' : done_reason,
                'done_type' : done_type, }

        return observation, reward, done, info

    def reset(self):
        """
        Resets the robot joint angle to [0,0,0,0,0,0]
        And Randomly initialize the position of the object
        """
        robot_success = self.robot.setJointPos([0,0,0,0,0,0], wait=True)
        # experience value
        x_limit = (-300, 300)
        y_limit = (-461, -300)
        if self.random_xy_type == 'gaussian':
            x_mm, y_mm = self.gaussian_xy(x_limit, y_limit)
        elif self.random_xy_type == 'uniform':
            x_mm, y_mm = self.uniform_xy(x_limit, y_limit)
        else:
            x_mm = random.randint(-400, 400)
            min_y_mm = -min(550, int((550 ** 2 - x_mm ** 2) ** 0.5))
            y_mm = random.randint(min_y_mm, -200)
        # set to 175 directly may cause object falling off the table
        z_mm = 175 + self.robot_base[2]
        obj_success = self._world.setModelPos("cube", [x_mm, y_mm, z_mm, 0, 0 ,0])
        time.sleep(0.1)
        self.object_pos = object_pos = self._world.getModelPos("cube")
        logger.info("Set Object Pos: %r ." % object_pos)
        # to make sure that object is on the table.
        if not obj_success or not robot_success or object_pos[2] < 175:
            logger.warn("Reset Failed !  Reset again. ")
            self.reset()

        # return obs !  
        self.observation = self._get_obs()
        return self.observation

    def _get_obs(self):
        """Returns the observation.
        """
        self.robot_pos = self.robot.getRobotPos()
        self.gripper_pos = [self.robot_pos[0], self.robot_pos[1], self.robot_pos[2]-GRIPPER_H_MM,
                            self.robot_pos[3], self.robot_pos[4], self.robot_pos[5],]
        img_left1 = self._cam_left1.getRGBImage()
        img_left2 = self._cam_left2.getRGBImage()
        # img_right1 = self._cam_right1.getRGBImage()
        # img_right2 = self._cam_right2.getRGBImage()
        obs = {'img_left1' : img_left1,
                'img_left2' : img_left2,
                'robot_pos' : self.robot_pos, 
                'gripper_pos' : self.gripper_pos} 
        return obs

    def _collision(self, action):
        """
        Judge if the robot's next positon can make collision by the Pose(z) of  robot and other objects.
        :param action: robot pose[x,y,z,u,v,w]
        """
        collision_z_mm = self.object_pos[2]-self.robot_base[2]+OBJECT_H_MM
        logger.info("Collision of Gripper to Object(Z): %f ." % collision_z_mm)
        if action[2] - GRIPPER_H_MM < collision_z_mm:
            logger.warn("May cause collision as robot gripper pos(z) is lower than the object pos(z)!")
            return True
        else:
            return False

    def _enter_grab_range(self):
        """
        judge if the robot reaches a position where it can grab objects.
        """
        xy_distance_mm = ((self.object_pos[0]-self.gripper_pos[0])**2 + \
                          (self.object_pos[1]-self.gripper_pos[1])**2)**0.5
        z_distance_mm = self.gripper_pos[2] - (self.object_pos[2]-self.robot_base[2]+OBJECT_H_MM)
        logger.info("Grab Distance : l2_xy: %f, Z : %f ." % (xy_distance_mm, z_distance_mm))
        if xy_distance_mm < OBJECT_W_MM/2 and z_distance_mm < self.grab_range_z_mm:
            return True
        else:
            return False

    def gazebo_control(self):
        return self.robot, self.gripper

    def gaussian_xy(self, x_limit, y_limit):
        """
        generate random (x,y) coordinates using Gaussian distribution;
        the mean of the Gaussian if set to the middle of the max - min range;
        the std of the Gaussian is set to a quarter of the max - min range;
        rejection sampling is used if the generated coordinates exceed either limit
        :param x_limit: a tuple (x_min, x_max) indicating the range in x-axis
        :param y_limit: a tuple (y_min, y_max) indicating the range in y-axis
        :return: a tuple (x,y)
        """
        x_min, x_max = x_limit
        y_min, y_max = y_limit
        assert x_min <= x_max, 'x_min <= x_max (%.4f <= %.4f)' % (x_min, x_max)
        assert y_min <= y_max, 'y_min <= y_max (%.4f <= %.4f)' % (y_min, y_max)

        x = np.random.randn() * (x_max - x_min) / 10 + (x_max + x_min) / 2
        y = np.random.randn() * (y_max - y_min) / 10 + (y_max + y_min) / 2
        if (not x_min <= x <= x_max) or (not y_min <= y <= y_max):
            x, y = self.gaussian_xy(x_limit, y_limit)
        return x, y
        # return (x_max + x_min) / 2, (y_max + y_min) / 2

    def uniform_xy(self, x_limit, y_limit):
        """
        generate random (x,y) coordinates using Uniform distribution
        :param x_limit: a tuple (x_min, x_max) indicating the range in x-axis
        :param y_limit: a tuple (y_min, y_max) indicating the range in y-axis
        :return: a tuple (x,y)
        """
        x_min, x_max = x_limit
        y_min, y_max = y_limit
        x = np.random.rand() * (x_max - x_min) + x_min
        y = np.random.rand() * (y_max - y_min) + y_min
        return x, y