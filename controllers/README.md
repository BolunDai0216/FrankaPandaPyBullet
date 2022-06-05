# Implemented Controllers

Here we provide a bried explanation for each of the implemented controller.

## Impedance Controller

The idea is to make the end-effector act like a spring-damper system. The end-effector force would have the form of a PD controller

$$F = \mathbf{K}_p\Delta\mathbf{p} + \mathbf{K}_d\Delta\mathbf{v}$$

Then, we have the joint torques being

$$\tau_F = J_F^TF$$

In addition we add two terms one for gravity compensation and the other to avoid the joint velocity being to large

$$\tau_G = C(\mathbf{q}, \dot{\mathbf{q}}) + G(\mathbf{q})\ \ \ \ \mathrm{and}\ \ \ \ \tau_v = -\mathbf{K}_d^\prime\dot{\mathbf{q}}$$

The actual torque that is applied to the manipulator would be

$$\tau = \tau_F + \tau_G + \tau_v.$$

## Feedback Controller

We can write the Taylor's expansion for the forward kinematic function as

$$\mathbf{x}^{\mathrm{des}} \approx \mathbf{F}(\mathbf{q}) + \frac{\partial\mathbf{F}}{\partial\mathbf{q}}\Big|_{\mathbf{q}}(\mathbf{q}^\prime - \mathbf{q}) + \epsilon = \mathbf{x} + \mathbf{J}(\mathbf{q})\Delta{\mathbf{q}} + \epsilon$$

this gives us

$$\Delta{\mathbf{q}} = \mathbf{J}^+(\mathbf{q})(\mathbf{x}^{\mathrm{des}} - \mathbf{x}).$$

Additionally, we have

$$\Delta{\dot{\mathbf{q}}} = \mathbf{J}^+(\mathbf{q})(\dot{\mathbf{x}}^{\mathrm{des}} - \dot{\mathbf{x}})$$

Similar to the case in impedance control we would also need to add terms for gravity compensation and joint velocity damping. This gives us the final controller as

$$\tau = \mathbf{K}_p\Delta{\mathbf{q}} + \mathbf{K}_d\Delta{\dot{\mathbf{q}}} + \tau_G + \tau_v.$$
