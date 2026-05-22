# ML_DM_Comp.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Car import Car
from ML import predict_optimal_pid, DirectPIDModel  

def get_target_speed(t):
    if t < 20:
        return 20.0
    elif t < 40:
        return 20.0 + (30.0 - 20.0) * (t - 20.0) / 20.0
    elif t < 60:
        return 30.0
    elif t < 62:
        return 30 - (30-15) * (t-60)/2
    elif t < 70:
        return 15.0
    elif t < 71.5:
        return 15 + (25 - 15) * (t - 70.0) / 1.5
    else:
        return 25.0

def smooth_step_turn(t, t0, t1, angle_deg):
    s = (t - t0) / (t1 - t0)
    s = np.clip(s, 0.0, 1.0)
    return np.radians(angle_deg) * 0.5 * (1 - np.cos(np.pi * s))

def get_target_heading(t):
    if t < 15:
        return 0.0

    # sharper ramp up to +20 deg
    elif t < 22:
        return smooth_step_turn(t, 15, 22, 20)

    # hold turn
    elif t < 28:
        return np.radians(20)

    # sharper transition to -18 deg
    elif t < 38:
        s = (t - 28) / (38 - 28)
        return np.radians(20 + (-18 - 20) * 0.5 * (1 - np.cos(np.pi * s)))

    # hold opposite turn
    elif t < 45:
        return np.radians(-18)

    # transition to +15 deg
    elif t < 55:
        s = (t - 45) / (55 - 45)
        return np.radians(-18 + (15 + 18) * 0.5 * (1 - np.cos(np.pi * s)))

    # hold
    elif t < 65:
        return np.radians(15)

    # return to zero
    elif t < 72:
        s = (t - 65) / (72 - 65)
        return np.radians(15 * 0.5 * (1 + np.cos(np.pi * s)))

    else:
        return 0.0

def print_compact(car):
    print("Car parameters:")
    print(" m    max_a  Iz     L     Cf     Cr     mu    Cd     A")
    print(
        f"{car.m:.0f}, {car.max_accel:.2f}, {car.Iz:.0f}, {car.L:.2f}, "
        f"{car.Cf:.0f}, {car.Cr:.0f}, {car.mu:.2f}, "
        f"{car.Cd:.2f}, {car.A:.2f}\n"
        )
    
    print("Slected PID values:")
    print(" p_s     i_s    d_s    p_v   i_v   d_v")
    print(
        f"{car.kp_steer:.2f}, {car.ki_steer:.2f}, {car.kd_steer:.2f}, "
        f"{car.kp_speed:.2f}, {car.ki_speed:.2f}, {car.kd_speed:.2f},\n"
    )

def plot_drive_conditions_stacked(times, states, target_speeds, wind_longs, wind_lats, road_grades, target_headings, psis):
    """
    Plot:
    1) Speed vs target speed
    2) Error over first 60 seconds
    3) Target heading
    4) Wind direction
    5) Road gradient

    All sharing the same time axis.
    """

    # ---------------------------
    # Compute signals
    # ---------------------------
    states = np.array(states)
    speed = np.linalg.norm(states[:, 3:5], axis=1)

    wind_direction = np.degrees(np.arctan2(wind_lats, wind_longs))
    road_gradient_deg = np.degrees(road_grades)
    target_heading_deg = np.degrees(target_headings)
    psis = np.degrees(np.unwrap(psis))

    # ---------------------------
    # Create figure
    # ---------------------------
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1,
        figsize=(14, 10),
        sharex=True
    )

    # ---------------------------
    # 1. Speed plot
    # ---------------------------
    ax1.plot(times, speed, label="Actual Speed [m/s]", linewidth=2)
    ax1.plot(times, target_speeds, label="Target Speed [m/s]", linewidth=2)

    ax1.set_ylabel("Speed [m/s]")
    ax1.set_title("Vehicle Performance and Disturbances")
    ax1.legend()
    ax1.grid(True)

    # ---------------------------
    # 2. Target heading
    # ---------------------------
    ax2.plot(times, psis, label="True heading [deg]", linewidth=2)
    ax2.plot(times, target_heading_deg, label="Target heading [deg]", linewidth=2)
    ax2.set_ylabel("Target Heading [deg]")
    ax2.legend()
    ax2.grid(True)

    # ---------------------------
    # 3. Wind direction
    # ---------------------------
    ax3.scatter(times, wind_direction, s=3)
    ax3.set_ylabel("Wind Direction [deg]")
    ax3.grid(True)

    # ---------------------------
    # 4. Road gradient
    # ---------------------------
    ax4.plot(times, road_gradient_deg, linestyle="-.", linewidth=2)
    ax4.set_ylabel("Road Gradient [deg]")
    ax4.set_xlabel("Time [s]")
    ax4.grid(True)

    # ---------------------------
    # Layout
    # ---------------------------
    plt.tight_layout()

rng = np.random.default_rng()

# -------------------------------------------------
# 1. Primary sampled parameters
# -------------------------------------------------
m = round(rng.uniform(800, 4500))        # small hatchback to loaded van
L = rng.uniform(2.2, 4.8)               # city car to large SUV  
mu = rng.uniform(0.3, 1.4)              # ice/gravel to dry tarmac
Cd = rng.uniform(0.18, 0.55)            # sports car to brick van
A = rng.uniform(1.6, 3.8)
max_accel = rng.uniform(0.8, 9.0)       # underpowered to sports
rho_air = 1.225

# Front/rear axle split as fraction of wheelbase
# Typical passenger cars sit around 45-55% front axle load distribution
front_weight_frac = rng.uniform(0.45, 0.60)

# Convert this into CG distances from front/rear axle
# Static load ratio:
# front axle load = m g (Lr/L)
# rear axle load  = m g (Lf/L)
Lr = front_weight_frac * L
Lf = L - Lr

# -------------------------------------------------
# 2. Derived yaw inertia
# -------------------------------------------------
# Rough scaling: Iz ~ k * m * L^2
# k for passenger vehicles often lands around 0.20-0.35
k_Iz = rng.uniform(0.22, 0.32)
Iz = round(k_Iz * m * L**2)

# -------------------------------------------------
# 3. Derived cornering stiffness
# -------------------------------------------------
# Use axle loads with a stiffness-per-load scaling.
# This is still simplified, but much better than independent random values.
g = 9.81
Fz_front = m * g * (Lr / L)
Fz_rear = m * g * (Lf / L)

# Cornering stiffness per axle load [1/rad]
# Chosen to land in a sensible passenger-car range.
c_alpha_scale_front = rng.uniform(7.0, 11.0)
c_alpha_scale_rear = rng.uniform(7.0, 11.0)

Cf = c_alpha_scale_front * Fz_front
Cr = c_alpha_scale_rear * Fz_rear

# -------------------------------------------------
# 4. Steering limits
# -------------------------------------------------
# Smaller wheelbase cars tend to steer a bit more sharply
base_max_steer_deg = np.clip(35.0 - 4.0 * (L - 2.4) / (3.2 - 2.4), 25.0, 45.0)
max_steer_angle = np.radians(base_max_steer_deg)

# Steering rate also varies, but keep it realistic
max_steer_rate = np.radians(90)

# -------------------------------------------------
# 5. Longitudinal capability
# -------------------------------------------------
# Use acceleration capability ranges typical of passenger vehicles
min_accel = -6  

# -------------------------------------------------
# Predict PID gains directly from car parameters
# using the two offline-trained DirectPIDModel instances
# (steer model: car params -> steering PID,
#  speed model: car params -> speed PID).
# No simulation or optimisation is run here.
# -------------------------------------------------
car_params = {
    "mass":                           m,
    "max_acceleration":               max_accel,
    "yaw_inertia":                    Iz,
    "length":                         L,
    "max_steering_stiffness_front":   Cf,
    "max_steering_stiffness_rear":    Cr,
    "tyre_friction":                  mu,
    "cd":                             Cd,
    "cross_sectional_area":           A,
}

result = predict_optimal_pid(
    car_params=car_params,
    steer_model_path="direct_steer_pid_model.pkl",
    speed_model_path="direct_speed_pid_model.pkl",
)

steer_kp = result["steer_kp"]
steer_ki = result["steer_ki"]
steer_kd = result["steer_kd"]
speed_kp = result["speed_kp"]
speed_ki = result["speed_ki"]
speed_kd = result["speed_kd"]

car = Car(
        m=m,
        Iz=Iz,
        L=L,
        Cf=Cf,
        Cr=Cr,
        mu=mu,
        Cd=Cd,
        A=A,
        rho_air=rho_air,
        max_steer_angle=max_steer_angle,
        max_steer_rate=max_steer_rate,
        max_accel=max_accel,
        min_accel=min_accel,
        kp_steer=steer_kp,
        ki_steer=steer_ki,
        kd_steer=steer_kd,
        kp_speed=speed_kp,
        ki_speed=speed_ki,
        kd_speed=speed_kd
    )

dt = 0.01
T = 90.0
N = int(T / dt)
N2 = 5

avg_speed_error = 0
avg_heading_error = 0

for _ in range(N2):
    car.reset_pid() 
    state = np.array([0.0, 0.0, 0.0, 20, 0.0, 0.0])

    states = []
    steers = []
    accels = []
    times = []

    # Retained disturbance histories
    target_headings = []
    target_speeds = []
    psis = []
    road_grades = []
    wind_longs = []
    wind_lats = []
    wind_yaws = []

    speed_err_accumulative = 0
    heading_err_accumulative = 0

    # ---------------------------
    # Disturbance initial values
    # ---------------------------
    # Disturbance states
    wind_long = 0.0
    wind_lat = 0.0
    wind_yaw = 0.0
    road_grade = 0.0
    lane_heading = 0.0

    # Per-simulation random amplitudes
    wind_amp_long = rng.uniform(0.2, 0.5)
    wind_amp_lat  = rng.uniform(0.2, 0.5)
    wind_amp_yaw  = rng.uniform(0.2, 0.5)
    road_amp = rng.uniform(0.2, 0.5)
    lane_amp = rng.uniform(0.2, 0.5)

    # Per-simulation random frequencies
    wind_freq_long = rng.uniform(0.2, 1.0)
    wind_freq_lat  = rng.uniform(0.2, 1.0)
    wind_freq_yaw  = rng.uniform(0.2, 1.0)
    road_freq = rng.uniform(0.02, 0.2)
    lane_freq = rng.uniform(0.02, 0.2)

    # Per-simulation random phases
    wind_phase_long = rng.uniform(0.0, 2 * np.pi)
    wind_phase_lat  = rng.uniform(0.0, 2 * np.pi)
    wind_phase_yaw  = rng.uniform(0.0, 2 * np.pi)
    road_phase = rng.uniform(0.0, 2 * np.pi)
    lane_phase = rng.uniform(0.0, 2 * np.pi)

    times_hp = np.arange(0, T + dt, dt)
    heading_profile = np.array([get_target_heading(times_hp[j]) for j in range(N)])
    target_yaw_rates = np.gradient(heading_profile, dt)

    for i in range(N):
        t = i * dt

        # Road gradient
        road_grade += road_amp * rng.normal(0.0, np.radians(0.03)) * dt
        road_grade += road_amp * np.radians(1.0) * np.sin(road_freq * t + road_phase) * dt
        road_grade = np.clip(road_grade, np.radians(-10), np.radians(10))

        # Wind
        wind_long += wind_amp_long * rng.normal(0.0, 0.2) * dt 
        wind_lat  += wind_amp_lat  * rng.normal(0.0, 0.6) * dt 
        wind_yaw  += wind_amp_yaw  * rng.normal(0.0, np.radians(0.8)) * dt

        wind_long += wind_amp_long * 0.1 * np.sin(wind_freq_long * t + wind_phase_long) * dt
        wind_lat  += wind_amp_lat  * 0.2 * np.sin(wind_freq_lat * t + wind_phase_lat) * dt
        wind_yaw  += wind_amp_yaw  * np.radians(1.5) * np.sin(wind_freq_yaw * t + wind_phase_yaw) * dt

        wind_long = np.clip(wind_long, -0.5, 0.5)
        wind_lat  = np.clip(wind_lat, -0.3, 0.3)
        wind_yaw  = np.clip(wind_yaw, -np.radians(8), np.radians(8))

        # Base desired speed
        target_speed = get_target_speed(t)

        # =========================================================
        # 3. Controlled vehicle step
        # =========================================================
        state, steer_cmd, accel_cmd = car.controlled_step(
            state=state,
            target_heading=get_target_heading(t),
            target_speed=get_target_speed(t),
            dt=dt
        )

        # =========================================================
        # 5. Apply disturbances to the new state
        # =========================================================
        x, y, psi, v, vy, r = state

        # Gravity effect from road slope
        # Positive road_grade = uphill => reduces forward speed
        a_grade = -9.81 * np.sin(road_grade)

        # Apply disturbed accelerations directly to the state
        v += (a_grade + wind_long) * dt
        vy += wind_lat * dt
        r += wind_yaw * dt

        psi += r * dt  
        # Update heading after yaw disturbance
        psi = car.wrap_angle(psi)

        # Prevent negative speed if desired
        v = max(v, 0.0)

        state = np.array([x, y, psi, v, vy, r])

        # =========================================================
        # 6. Performance metric
        # =========================================================
        target_yaw_rate = target_yaw_rates[i]
        actual_yaw_rate = state[5]  # r is index 5 in [x, y, psi, v, vy, r]

        speed_err    = abs(target_speed - np.linalg.norm([v, vy]))
        heading_err  = abs(car.wrap_angle(psi - get_target_heading(t)))
        yaw_rate_err = abs(actual_yaw_rate - target_yaw_rate)

        speed_err_accumulative   += speed_err * dt
        heading_err_accumulative += (heading_err * 8.0 + yaw_rate_err * 5.0) * dt

        # =========================================================
        # 7. Store results
        # =========================================================
        states.append(state.copy())
        steers.append(steer_cmd)
        accels.append(accel_cmd)
        times.append(t)

        # Store disturbance history
        target_headings.append(get_target_heading(t))
        target_speeds.append(get_target_speed(t))
        psis.append(psi)
        road_grades.append(road_grade)
        wind_longs.append(wind_long)
        wind_lats.append(wind_lat)
        wind_yaws.append(wind_yaw)

    avg_speed_error += speed_err_accumulative / N2
    avg_heading_error += heading_err_accumulative / N2

plot_drive_conditions_stacked(
    times=times,
    states=states,
    target_speeds=target_speeds,
    wind_longs=wind_longs,
    wind_lats=wind_lats,
    road_grades=road_grades,
    target_headings=target_headings,
    psis=psis
)
        

print("\n")
print_compact(car)
print(f"Dynamic Model acculumative error = {avg_speed_error:.2f}, {avg_heading_error:.2f}\n")
plt.show()
