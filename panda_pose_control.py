import copy
import pathlib
from pdb import set_trace

import numpy as np
import numpy.linalg as LA
import pinocchio as pin
import pybullet as p
import pybullet_data
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R

from controllers.utils import get_state_update_pinocchio, send_joint_command


def alpha_func(t, T=5.0):
    if t <= T:
        alpha = np.sin((np.pi / 4) * (1 - np.cos(np.pi * t / T)))
        dalpha = (
            ((np.pi ** 2) / (4 * T))
            * np.cos((np.pi / 4) * (1 - np.cos(np.pi * t / T)))
            * np.sin(np.pi * t / T)
        )
    else:
        alpha = 1.0
        dalpha = 0.0

    return alpha, dalpha


def axis_angle_from_rot_mat(rot_mat):
    rotation = R.from_matrix(rot_mat)
    axis_angle = rotation.as_rotvec()
    angle = LA.norm(axis_angle)
    axis = axis_angle / angle

    return axis, angle


def main():
    """
    This file implements an operational space controller proportional controller.

    The part of the controller that controls the position is:
    τ_p = J_p^+[K_{p, p}(x_des - x)]

    The part of the controller that controls the orientation is:
    τ_o = J_o^+[K_{p, φ}(φ_des - φ)]

    where φ = θr, with θ being the angle in the axis-angle representation and
    r being the axis of rotation.

    Then, the final controller is:
    τ = τ_p + τ_o + G(q)
    """
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

    p_end = np.array([[-0.3], [-0.4], [0.5]])

    # get initial rotation and position
    q, dq = get_state_update_pinocchio(robot, robotID)
    R_start = copy.deepcopy(robot.data.oMf[FRAME_ID].rotation)
    _p_start = robot.data.oMf[FRAME_ID].translation
    p_start = _p_start[:, np.newaxis]

    # Get target orientation based on initial orientation
    _R_end = (
        R.from_euler("x", 45, degrees=True).as_matrix()
        @ R.from_euler("y", -40, degrees=True).as_matrix()
        @ R_start
    )
    R_end = R.from_matrix(_R_end).as_matrix()
    R_error = R_end @ R_start.T
    axis_error, angle_error = axis_angle_from_rot_mat(R_error)

    for i in range(80000):
        # Update pinocchio model and get joint states
        q, dq = get_state_update_pinocchio(robot, robotID)

        # Get simulation time
        sim_time = i * (1 / 240)

        # Compute α and dα
        alpha, dalpha = alpha_func(sim_time, T=30.0)

        # Compute p_target
        p_target = p_start + alpha * (p_end - p_start)

        # Compute v_target
        v_target = dalpha * (p_end - p_start)

        # Compute R_target
        theta_t = alpha * angle_error
        R_target = R.from_rotvec(axis_error * theta_t).as_matrix() @ R_start

        # Compute ω_target
        ω_target = dalpha * axis_error * angle_error

        # Get end-effector position
        _p_current = robot.data.oMf[FRAME_ID].translation
        p_current = _p_current[:, np.newaxis]

        # Get end-effector orientation
        R_current = robot.data.oMf[FRAME_ID].rotation

        # Error rotation matrix
        R_err = R_target @ R_current.T

        # Orientation error in axis-angle form
        rotvec_err = R.from_matrix(R_err).as_rotvec()

        # Get frame ID for grasp target
        jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        # Get Jacobian from grasp target frame
        # preprocessing is done in get_state_update_pinocchio()
        jacobian = robot.getFrameJacobian(FRAME_ID, jacobian_frame)

        # Get pseudo-inverse of frame Jacobian
        pinv_jac = np.linalg.pinv(jacobian)

        # Compute Gravitational terms
        G = robot.gravity(q)

        # Compute controller
        delta_x = np.zeros((6, 1))
        delta_x[:3] = p_target - p_current
        delta_x[3:] = rotvec_err[:, np.newaxis]
        delta_q = pinv_jac @ delta_x

        dx = np.zeros((6, 1))
        dx[:3] = v_target
        dx[3:] = ω_target[:, np.newaxis]
        delta_dq = pinv_jac @ dx - dq[:, np.newaxis]
        Kp = 3 * np.eye(9)
        Kd = np.eye(9)

        tau = (Kp @ delta_q + Kd @ delta_dq) + G[:, np.newaxis]

        # Set control for the two fingers to zero
        tau[-1] = 0.0
        tau[-2] = 0.0

        # Send joint commands to motor
        send_joint_command(robotID, tau)

        if i % 500 == 0:
            print(
                "Iter {:.2e} \t ǁeₒǁ₂: {:.2e} \t ǁeₚǁ₂: {:.2e}".format(
                    i, LA.norm(rotvec_err), LA.norm(p_target - p_current)
                ),
            )

        p.stepSimulation()

    p.disconnect()


if __name__ == "__main__":
    main()
