import numpy as np


class Car:
    def __init__(self,
                m, Iz, L,
                Cf, Cr,
                mu,
                Cd, A, rho_air,
                max_steer_angle,
                max_steer_rate,
                max_accel,
                min_accel,
                kp_steer, ki_steer, kd_steer,
                kp_speed, ki_speed, kd_speed):

        # ---------------------------
        # Vehicle parameters
        # ---------------------------
        self.m = m
        self.Iz = Iz
        self.L = L
        self.Cf = Cf
        self.Cr = Cr
        self.mu = mu
        self.Cd = Cd
        self.A = A
        self.rho_air = rho_air

        # Assume CG halfway between axles
        self.Lf = L / 2
        self.Lr = L / 2

        # ---------------------------
        # Actuator limits
        # ---------------------------
        self.max_steer_angle = max_steer_angle
        self.max_steer_rate = max_steer_rate
        self.max_accel = max_accel
        self.min_accel = min_accel

        # ---------------------------
        # Controller gains
        # ---------------------------
        self.kp_steer = kp_steer
        self.ki_steer = ki_steer
        self.kd_steer = kd_steer

        self.kp_speed = kp_speed
        self.ki_speed = ki_speed
        self.kd_speed = kd_speed

        # ---------------------------
        # Internal actuator state
        # ---------------------------
        self.delta = 0.0  # current steering angle [rad]

        # ---------------------------
        # PID memory terms
        # ---------------------------
        self.steer_integral = 0.0
        self.steer_prev_error = 0.0

        self.speed_integral = 0.0
        self.speed_prev_error = 0.0

    @staticmethod
    def wrap_angle(angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def reset_pid(self):
        """Reset PID integrators and previous errors."""
        self.steer_integral = 0.0
        self.steer_prev_error = 0.0
        self.speed_integral = 0.0
        self.speed_prev_error = 0.0

    def step_dynamics(self, state, steer_cmd, accel_cmd, dt):
        """
        Advance the car dynamics by one timestep.

        Parameters
        ----------
        state : array-like
            [x, y, psi, v, vy, r]
            x   : global x-position [m]
            y   : global y-position [m]
            psi : heading/yaw angle [rad]
            v   : longitudinal velocity [m/s]
            vy  : lateral velocity [m/s]
            r   : yaw rate [rad/s]

        steer_cmd : float
            Desired steering angle [rad]

        accel_cmd : float
            Desired longitudinal acceleration [m/s^2]

        dt : float
            Timestep [s]
        """

        x, y, psi, v, vy, r = state

        # ---------------------------
        # Steering actuator dynamics
        # ---------------------------
        steer_cmd = np.clip(
            steer_cmd,
            -self.max_steer_angle,
            self.max_steer_angle
        )

        steer_error = steer_cmd - self.delta
        max_delta_change = self.max_steer_rate * dt
        steer_change = np.clip(
            steer_error,
            -max_delta_change,
            max_delta_change
        )

        self.delta += steer_change
        self.delta = np.clip(
            self.delta,
            -self.max_steer_angle,
            self.max_steer_angle
        )

        delta = self.delta

        # Avoid division by zero
        v_safe = max(abs(v), 0.1)

        # ---------------------------
        # Slip angles
        # ---------------------------
        alpha_f = (vy + self.Lf * r) / v_safe - delta
        alpha_r = (vy - self.Lr * r) / v_safe

        # ---------------------------
        # Linear tyre model
        # ---------------------------
        Fyf = -self.Cf * alpha_f
        Fyr = -self.Cr * alpha_r

        # Friction-limited lateral force per axle
        F_max = self.mu * self.m * 9.81 / 2.0
        Fyf = np.clip(Fyf, -F_max, F_max)
        Fyr = np.clip(Fyr, -F_max, F_max)

        # ---------------------------
        # Aerodynamic drag force
        # ---------------------------
        drag_force = 0.5 * self.rho_air * self.Cd * self.A * v * abs(v)

        # ---------------------------
        # Equations of motion
        # ---------------------------
        dv = accel_cmd - drag_force / self.m
        dvy = (Fyf + Fyr) / self.m - v * r
        dr = (self.Lf * Fyf - self.Lr * Fyr) / self.Iz

        # ---------------------------
        # Integrate body states
        # ---------------------------
        v += dv * dt
        vy += dvy * dt
        r += dr * dt
        psi += r * dt
        psi = self.wrap_angle(psi)

        # ---------------------------
        # Global position update
        # ---------------------------
        x_dot = v * np.cos(psi) - vy * np.sin(psi)
        y_dot = v * np.sin(psi) + vy * np.cos(psi)

        x += x_dot * dt
        y += y_dot * dt

        return np.array([x, y, psi, v, vy, r])

    def pid_control(self, state, target_heading, target_speed, dt):
        """
        Compute steering and acceleration commands from PID controllers.

        Parameters
        ----------
        state : array-like
            [x, y, psi, v, vy, r]
        target_heading : float
            Desired heading angle [rad]
        target_speed : float
            Desired longitudinal speed [m/s]
        dt : float
            Timestep [s]
        """

        _, _, psi, v, vy, _ = state

        # ---------------------------
        # Steering PID (heading control)
        # ---------------------------
        heading_error = self.wrap_angle(target_heading - psi)

        self.steer_integral += heading_error * dt
        heading_error_derivative = (heading_error - self.steer_prev_error) / dt

        steer_cmd = (
            self.kp_steer * heading_error
            + self.ki_steer * self.steer_integral
            + self.kd_steer * heading_error_derivative
        )

        self.steer_prev_error = heading_error

        # Clamp steering command
        steer_cmd = np.clip(steer_cmd,
                            -self.max_steer_angle,
                            self.max_steer_angle)

        # Optional simple anti-windup
        if abs(steer_cmd) >= self.max_steer_angle:
            self.steer_integral -= heading_error * dt

        # ---------------------------
        # Speed PID
        # ---------------------------
        speed_error = target_speed - np.linalg.norm([v, vy])

        self.speed_integral += speed_error * dt
        speed_error_derivative = (speed_error - self.speed_prev_error) / dt

        accel_cmd = (
            self.kp_speed * speed_error
            + self.ki_speed * self.speed_integral
            + self.kd_speed * speed_error_derivative
        )

        self.speed_prev_error = speed_error

        accel_cmd = np.clip(accel_cmd, self.min_accel, self.max_accel)

        # Optional anti-windup
        if accel_cmd <= self.min_accel or accel_cmd >= self.max_accel:
            self.speed_integral -= speed_error * dt

        return steer_cmd, accel_cmd

    def controlled_step(self, state, target_heading, target_speed, dt):
        """
        One full closed-loop step:
        1. use PID to compute commands
        2. advance vehicle dynamics

        Returns
        -------
        new_state : np.ndarray
        steer_cmd : float
        accel_cmd : float
        """
        steer_cmd, accel_cmd = self.pid_control(
            state=state,
            target_heading=target_heading,
            target_speed=target_speed,
            dt=dt
        )

        new_state = self.step_dynamics(
            state=state,
            steer_cmd=steer_cmd,
            accel_cmd=accel_cmd,
            dt=dt
        )

        return new_state, steer_cmd, accel_cmd
