import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from matplotlib.ticker import MultipleLocator


class FeedbackController:
    def __init__(self, init_pos, target_pos, terminal_time):
        self.init_pos = init_pos
        self.target_pos = target_pos
        self.T = terminal_time

        self.target_dis = np.linalg.norm(self.target_pos - self.init_pos)
        self.target_vec = (self.target_pos - self.init_pos) / self.target_dis

        # Get coefficients of the planned trajectory
        self.plan_trajectory()

    def plan_trajectory(self):
        A_mat = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [self.T**5, self.T**4, self.T**3, self.T**2, self.T, 1.0],
                [
                    5 * self.T**4,
                    4 * self.T**3,
                    3 * self.T**2,
                    2 * self.T,
                    1.0,
                    0.0,
                ],
                [
                    20 * self.T**3,
                    12 * self.T**2,
                    6 * self.T,
                    2.0,
                    0.0,
                    0.0,
                ],
            ]
        )

        B_Vec = np.array([[0.0], [0.0], [0.0], [self.target_dis], [0.0], [0.0]])
        self.coeffs = np.linalg.inv(A_mat) @ B_Vec

    def retrieve_plan(self, t):
        if t <= self.T:
            time_vec_pos = np.array(
                [[t**5], [t**4], [t**3], [t**2], [t**1], [0.0]]
            )
            time_vec_vel = np.array(
                [[5 * t**4], [4 * t**3], [3 * t**2], [2 * t**1], [1.0], [0.0]]
            )
            time_vec_acc = np.array(
                [[20 * t**3], [12 * t**2], [6 * t**1], [2.0], [0.0], [0.0]]
            )

            pos = self.coeffs.T @ time_vec_pos * self.target_vec + self.init_pos
            vel = self.coeffs.T @ time_vec_vel * self.target_vec
            acc = self.coeffs.T @ time_vec_acc * self.target_vec

        else:
            pos = self.target_pos
            vel = 0.0 * self.target_vec
            acc = 0.0 * self.target_vec

        return pos, vel, acc

    def save_plan(self, name="imgs/feedforward_plan.png"):
        N = 200
        times = np.linspace(0, self.T + 1, N)

        pos_list = []
        vel_list = []
        acc_list = []

        for i in range(N):
            _p, _v, _a = self.retrieve_plan(times[i])
            pos_list.append(_p)
            vel_list.append(_v)
            acc_list.append(_a)

        pos_arr = np.concatenate(pos_list, axis=1)
        vel_arr = np.concatenate(vel_list, axis=1)
        acc_arr = np.concatenate(acc_list, axis=1)
        arrs = [pos_arr, vel_arr, acc_arr]
        titles = ["X", "Y", "Z"]
        ylabels = ["Position [m]", "Velocity [m/s]", "Acceleration [m/s2]"]

        fig, axs = plt.subplots(3, 3, figsize=(8, 8), sharex=True, sharey="row")
        for i in range(3):
            for j in range(3):
                axs[i][j].plot(
                    times, arrs[i][j, :], linewidth=4, color="cornflowerblue"
                )
                axs[i][j].xaxis.set_major_locator(MultipleLocator(2.0))
                axs[i][j].xaxis.set_minor_locator(MultipleLocator(0.5))

                # These need to be changed for different configurations
                if i == 0:
                    axs[i][j].yaxis.set_major_locator(MultipleLocator(0.4))
                    axs[i][j].yaxis.set_minor_locator(MultipleLocator(0.1))
                if i == 1:
                    axs[i][j].yaxis.set_major_locator(MultipleLocator(0.05))
                    axs[i][j].yaxis.set_minor_locator(MultipleLocator(0.0125))
                if i == 2:
                    axs[i][j].yaxis.set_major_locator(MultipleLocator(0.06))
                    axs[i][j].yaxis.set_minor_locator(MultipleLocator(0.015))

                axs[i][j].grid(True, "minor", color="0.85", linewidth=0.50, zorder=-20)
                axs[i][j].grid(True, "major", color="0.65", linewidth=0.75, zorder=-10)
                axs[i][j].tick_params(which="both", bottom=False, left=False)

                if i == 0:
                    axs[i][j].set_title(titles[j])
                if i == 2:
                    axs[i][j].set_xlabel("Time [s]")
                if j == 0:
                    axs[i][j].set_ylabel(ylabels[i])
                    axs[i][j].yaxis.set_label_coords(-0.3, 0.5)

        plt.tight_layout()
        plt.savefig(name, dpi=200, transparent=False, bbox_inches="tight")

    def get_control(self, robot, t, q, dq, frameID):
        # Get planned position, velocity and accleration
        pos, vel, acc = self.retrieve_plan(t)

        # Get frame ID for grasp target
        jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        # Get Jacobian from grasp target frame
        jacobian = robot.getFrameJacobian(frameID, jacobian_frame)

        # Get frame position and velocity
        position = robot.data.oMf[frameID].translation
        velocity = jacobian[:3, :] @ dq[:, np.newaxis]

        # Get frame error
        delta_p = pos - position[:, np.newaxis]
        delta_v = vel - velocity

        # Get pseudo-inverse of frame Jacobian
        pinv_jac = np.linalg.pinv(jacobian[:3, :])

        # Compute Coriolis and Gravitational terms
        C = robot.nle(q, dq)

        # Compute torque
        tau = (
            C[:, np.newaxis]
            + pinv_jac @ delta_p
            + 0.5 * pinv_jac @ delta_v
            - 0.05 * (np.eye(9) @ dq[:, np.newaxis])
        )

        return tau
