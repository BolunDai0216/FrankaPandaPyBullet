import copy
import pathlib
import time
from pdb import set_trace

import numpy as np
import pinocchio as pin
import pybullet as p
import pybullet_data
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R

from controllers.utils import (
    compute_quat_vec_error,
    get_state_update_pinocchio,
    send_joint_command,
)

MODE_ROTATE = 1
MODE_STATIC = 2


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
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
    init = True

    last_quat_err = np.array([0.0, 0.0, 0.0, 1.0])

    while True:
        # Update pinocchio model and get joint states
        q, dq = get_state_update_pinocchio(robot, robotID)

        # Get end-effector position
        _gt_position = robot.data.oMf[FRAME_ID].translation
        gt_position = _gt_position[:, np.newaxis]

        if init:
            _init_rotation = robot.data.oMf[FRAME_ID].rotation  # R10
            _target_rotation = (
                R.from_euler("z", 90, degrees=True).as_matrix() @ _init_rotation
            )  # R20
            target_rotation = R.from_matrix(_target_rotation)
            target_quaternion = copy.deepcopy(target_rotation.as_quat())
            target_position = np.array([[0.3], [0.4], [0.5]])
            init = False

        # Get end-effector orientation
        _gt_orientation = robot.data.oMf[FRAME_ID].rotation

        # Error rotation matrix
        R_err = target_rotation.as_matrix() @ _gt_orientation.T
        quat_err = R.from_matrix(R_err).as_quat()

        cond = np.dot(target_quaternion, quat_err)
        # cond = np.dot(last_quat_err, quat_err)
        if cond < 0:
            quat_err = -quat_err
        last_quat_err = quat_err

        # Get frame ID for grasp target
        jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        # Get Jacobian from grasp target frame
        # preprocessing is done in get_state_update_pinocchio()
        jacobian = robot.getFrameJacobian(FRAME_ID, jacobian_frame)

        # Get pseudo-inverse of frame Jacobian
        pinv_jac = np.linalg.pinv(jacobian)
        pinv_jac[-1, :] = 0.0
        pinv_jac[-2, :] = 0.0

        # Compute Gravitational terms
        G = robot.gravity(q)

        # get velocity part of quat_err
        quat_err_vel = quat_err[:3, np.newaxis]

        if mode == MODE_ROTATE:
            target_dx = np.zeros((6, 1))
            target_dx[:3] = 1.0 * (target_position - gt_position)
            target_dx[3:] = -np.diag([0.5, 0.5, 0.5]) @ quat_err_vel
            tau = (pinv_jac @ target_dx - dq[:, np.newaxis]) + G[:, np.newaxis]

            # if np.linalg.norm(quat_err_vel) <= 4.4e-2:
            #     mode = MODE_STATIC
            #     target_joint = q

        elif mode == MODE_STATIC:
            tau = 0.5 * (target_joint - q) + 0.5 * (0 - dq) + G

        tau[-1] = 0.0
        tau[-2] = 0.0

        print(quat_err_vel, target_position - gt_position)

        # Send joint commands to motor
        send_joint_command(robotID, tau)

        p.stepSimulation()

    p.disconnect()


if __name__ == "__main__":
    main()
