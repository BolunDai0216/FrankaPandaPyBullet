import pathlib

import numpy as np
import pinocchio as pin
import pybullet as p
import pybullet_data
from pinocchio.robot_wrapper import RobotWrapper

from controllers.utils import get_state_update_pinocchio, send_joint_command

MODE_ROTATE = 1
MODE_STATIC = 2


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # Load plane
    p.loadURDF("plane.urdf")

    # Load Franka Panda Robot
    file_directory = str(pathlib.Path(__file__).parent.resolve())
    robot_URDF = file_directory + "/franka_panda/panda.urdf"
    robotID = p.loadURDF(robot_URDF, useFixedBase=True)

    # Build pin_robot
    robot = RobotWrapper.BuildFromURDF(robot_URDF)

    # Get frame ID for grasp target
    FRAME_ID = robot.model.getFrameId("panda_grasptarget")

    # Get active joint ids
    active_joint_ids = [0, 1, 2, 3, 4, 5, 6, 10, 11]

    # Disable the velocity control on the joints as we use torque control.
    p.setJointMotorControlArray(
        robotID, active_joint_ids, p.VELOCITY_CONTROL, forces=np.zeros(9),
    )

    target_joint_angles = [
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

    for i, joint_ang in enumerate(target_joint_angles):
        p.resetJointState(robotID, active_joint_ids[i], joint_ang, 0.0)

    mode = MODE_ROTATE

    while True:
        # Update pinocchio model and get joint states
        q, dq = get_state_update_pinocchio(robot, robotID)

        # Get frame ID for grasp target
        jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        # Get Jacobian from grasp target frame
        # preprocessing is done in get_state_update_pinocchio()
        jacobian = robot.getFrameJacobian(FRAME_ID, jacobian_frame)

        # Get pseudo-inverse of frame Jacobian
        pinv_jac = np.linalg.pinv(jacobian)

        # Compute Gravitational terms
        G = robot.gravity(q)

        if mode == MODE_ROTATE:
            target_dx = np.array(
                [[0.00], [0.0], [0.0], [0.0], [0.0], [0.1 * (q[6] - 2.0)]]
            )

            tau = 0.5 * (pinv_jac @ target_dx - dq[:, np.newaxis]) + G[:, np.newaxis]

            if np.linalg.norm(q[6] - 2.0) <= 1e-3:
                mode = MODE_STATIC
                target_joint = q

        elif mode == MODE_STATIC:
            tau = 0.5 * (target_joint - q) + 0.5 * (0 - dq) + G

        # Send joint commands to motor
        send_joint_command(robotID, tau)

        p.stepSimulation()

    p.disconnect()


if __name__ == "__main__":
    main()
