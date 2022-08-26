import pathlib

import numpy as np
import pybullet as p
import pybullet_data
from pinocchio.robot_wrapper import RobotWrapper

from controllers.utils import get_state_update_pinocchio, send_joint_command


def compute_quat_vec_error(quat_1, quat_2):
    eta1 = quat_1[0]
    eta2 = quat_2[0]

    quat_vec1 = quat_1[1:]
    quat_vec2 = quat_2[1:]

    delta_quat_vec = (
        eta1 * quat_vec2 - eta2 * quat_vec1 - np.cross(quat_vec1, quat_vec2)
    )

    return delta_quat_vec


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    # p.setTimeStep(1 / 240)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # Load plane
    p.loadURDF("plane.urdf")

    # Load Franka Panda Robot
    file_directory = str(pathlib.Path(__file__).parent.resolve())
    robot_URDF = file_directory + "/franka_panda/panda.urdf"
    robotID = p.loadURDF(robot_URDF, useFixedBase=True)

    # Build pin_robot
    robot = RobotWrapper.BuildFromURDF(robot_URDF)

    # Get active joint ids
    active_joint_ids = [0, 1, 2, 3, 4, 5, 6, 10, 11]

    # Disable the velocity control on the joints as we use torque control.
    p.setJointMotorControlArray(
        robotID, active_joint_ids, p.VELOCITY_CONTROL, forces=np.zeros(9),
    )

    target_joint = np.array(
        [
            0.0,
            -0.785398163,
            0.0,
            -2.35619449,
            0.0,
            1.57079632679,
            0.785398163397,
            0.001,
            0.001,
        ]
    )

    while True:
        # Update pinocchio model and get joint states
        q, dq = get_state_update_pinocchio(robot, robotID)

        # Compute Gravitational terms
        G = robot.gravity(q)

        tau = 0.5 * (target_joint - q) + 0.5 * (0 - dq) + G

        # Send joint commands to motor
        send_joint_command(robotID, tau[:, np.newaxis])

        p.stepSimulation()

    p.disconnect()


if __name__ == "__main__":
    main()
