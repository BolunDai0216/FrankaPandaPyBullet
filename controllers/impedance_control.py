import numpy as np
import pinocchio as pin


def impedance_control(robot, frameID, pos_des, q, dq, tol=5e-3):
    # Define impedance parameters
    Kp = np.diag([1.0, 1.0, 1.0])
    Kd = np.diag([0.5, 0.5, 0.5])

    # Define frame which Jacobian is computed
    jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

    # Get current position of frame
    position = robot.data.oMf[frameID].translation

    # Get frame Jacobian
    jacobian = robot.getFrameJacobian(frameID, jacobian_frame)

    # Get velocity of frame
    velocity = jacobian[:3, :] @ dq[:, np.newaxis]

    # Get gravity compensation torques
    C = robot.nle(q, dq)

    # f = Kp * (x_des - x) + Kd * (v_des - v)
    f = Kp @ (pos_des - position)[:, np.newaxis] - Kd @ (velocity)

    if np.linalg.norm(pos_des - position) < tol:
        tau = C + np.zeros(9) - 0.05 * (np.eye(9) @ dq[:, np.newaxis])[:, 0]
    else:
        # Get corresponding joint torque
        tau = (
            C
            + (jacobian[:3, :].T @ f)[:, 0]
            - 0.05 * (np.eye(9) @ dq[:, np.newaxis])[:, 0]
        )

    return tau
