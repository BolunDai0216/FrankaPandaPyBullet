import pathlib

import numpy as np
import pybullet as p
import pybullet_data
from gym import Env, spaces
from pinocchio.robot_wrapper import RobotWrapper

from controllers.utils import get_state_update_pinocchio, send_joint_command


class PandaEnv(Env):
    def __init__(self, render=True):
        super().__init__()

        if render:
            self.client = p.connect(p.GUI)
            # Improves rendering performance on M1 Macs
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)

        # Load plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # Load Franka Panda Robot
        file_directory = str(pathlib.Path(__file__).parent.resolve())
        robot_URDF = file_directory + "/franka_panda/panda.urdf"
        self.robotID = p.loadURDF(robot_URDF, useFixedBase=True)

        # Build pin_robot
        self.robot = RobotWrapper.BuildFromURDF(robot_URDF)

        # Get frame ID for grasp target
        self.FRAME_ID = self.robot.model.getFrameId("panda_grasptarget")

        # Get active joint ids
        self.active_joint_ids = [0, 1, 2, 3, 4, 5, 6, 10, 11]

        # Disable the velocity control on the joints as we use torque control.
        p.setJointMotorControlArray(
            self.robotID, self.active_joint_ids, p.VELOCITY_CONTROL, forces=np.zeros(9),
        )

        # Get number of joints
        self.n_j = p.getNumJoints(self.robotID)

        # Set observation and action space
        obs_low_q = []
        obs_low_dq = []
        obs_high_q = []
        obs_high_dq = []
        _act_low = []
        _act_high = []

        for i in range(self.n_j):
            _joint_infos = p.getJointInfo(self.robotID, i)  # get info of each joint

            if _joint_infos[2] != p.JOINT_FIXED:
                obs_low_q.append(_joint_infos[8])
                obs_high_q.append(_joint_infos[9])
                obs_low_dq.append(-_joint_infos[11])
                obs_high_dq.append(_joint_infos[11])
                _act_low.append(-_joint_infos[10])
                _act_high.append(_joint_infos[10])

        obs_low = np.array(obs_low_q + obs_low_dq, dtype=np.float32)
        obs_high = np.array(obs_high_q + obs_high_dq, dtype=np.float32)
        act_low = np.array(_act_low, dtype=np.float32)
        act_high = np.array(_act_high, dtype=np.float32)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

    def reset(self):
        q, dq = get_state_update_pinocchio(self.robot, self.robotID)
        state = np.concatenate((q, dq))

        return state

    def step(self, action):
        send_joint_command(self.robotID, action)
        p.stepSimulation()

        q, dq = get_state_update_pinocchio(self.robot, self.robotID)
        state = np.concatenate((q, dq))

        return state

    def close(self):
        p.disconnect()
