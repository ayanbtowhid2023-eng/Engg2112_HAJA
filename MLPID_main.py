"""
Offline Optimal PID Dataset Generator
=====================================

This file replaces the old random-PID dataset generation method.

Old method:
    random car + random PID -> speed_error, heading_error

New method:
    random car -> optimise PID offline -> save optimal PID gains

The output CSV is then used by ML.py to train:

    car parameters -> optimal steering PID
    car parameters -> optimal speed PID

This file is intentionally optimisation-heavy. It is meant to be run offline.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt

from Car import Car
from ML import CAR_PARAM_COLS, train_direct_pid_models


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

OUTPUT_CSV = "optimal_pid_dataset.csv"

NUM_CARS = 500     
N_PROFILE_RUNS = 5    
DT = 0.02        
T = 90                  
N = int(T / DT)

RNG = np.random.default_rng(42)

def generate_reference_profile(N, dt, rng):
    """
    Generate a seeded target speed and heading profile.
    Structure: alternating transition and hold segments with an injected
    double lane change to stress lateral dynamics.
    """
    times = np.arange(N) * dt

    def smooth_interp(t, t0, t1, v0, v1):
        s = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
        return v0 + (v1 - v0) * 0.5 * (1 - np.cos(np.pi * s))

    # --- Speed profile ---
    speeds = [
        rng.uniform(15.0, 25.0),
        rng.uniform(30, 40),
        rng.uniform(10.0, 20.0),
        rng.uniform(20.0, 30.0),
    ]
    hold_dur  = [rng.uniform(12.0, 20.0) for _ in range(4)]
    trans_dur = [rng.uniform(2.0,   6.0) for _ in range(3)]

    speed_segments = []
    t = 0.0
    for i in range(len(speeds)):
        speed_segments.append((t, t + hold_dur[i], 'hold', speeds[i], speeds[i]))
        t += hold_dur[i]
        if i < len(speeds) - 1:
            speed_segments.append((t, t + trans_dur[i], 'ramp', speeds[i], speeds[i+1]))
            t += trans_dur[i]

    speed_profile = np.zeros(N)
    for i in range(N):
        ti = times[i]
        val = speeds[0]
        for (t0, t1, kind, v0, v1) in speed_segments:
            if t0 <= ti < t1:
                val = v0 if kind == 'hold' else smooth_interp(ti, t0, t1, v0, v1)
                break
            val = speed_segments[-1][3]
        speed_profile[i] = val

    # --- Heading profile ---
    # Fast transitions (1.0-2.5s) stress lateral dynamics by demanding
    # high yaw rates — directly exposing Iz, Cf, Cr differences between cars
    max_angle   = rng.uniform(8.0,  15.0)   # was 10-22

    angles_deg = [0.0]
    for _ in range(4):
        sign = rng.choice([-1, 1])
        angles_deg.append(float(sign * rng.uniform(5.0, max_angle)))
    angles_deg.append(0.0)

    h_hold_dur  = [rng.uniform(8.0,  14.0) for _ in range(len(angles_deg))]
    h_trans_dur = [rng.uniform(2.0, 4.0) for _ in range(len(angles_deg) - 1)] 

    heading_segments = []
    t = 0.0
    for i in range(len(angles_deg)):
        if t == 0:
            h_hold_dur[i] = 20
        heading_segments.append((t, t + h_hold_dur[i], 'hold',
                                  np.radians(angles_deg[i]),
                                  np.radians(angles_deg[i])))
        t += h_hold_dur[i]
        if i < len(angles_deg) - 1:
            heading_segments.append((t, t + h_trans_dur[i], 'ramp',
                                      np.radians(angles_deg[i]),
                                      np.radians(angles_deg[i+1])))
            t += h_trans_dur[i]

    # --- Inject double lane change ---
    dlc_start = 1
    dlc_angle = rng.uniform(8.0, 15.0)

    # Set duration based on a conservative yaw rate — achievable by all cars
    # Use 0.15 rad/s as the minimum yaw rate any car should manage
    min_yaw_rate = 0.15  # rad/s
    dlc_duration = np.radians(dlc_angle) / min_yaw_rate * 3 # time to complete each phase

    dlc_segments = [
        (dlc_start, dlc_start +   dlc_duration, 'ramp', 0.0, np.radians( dlc_angle)),
        (dlc_start + dlc_duration, dlc_start + 2*dlc_duration, 'ramp', np.radians( dlc_angle), np.radians(-dlc_angle)),
        (dlc_start + 2*dlc_duration, dlc_start + 3*dlc_duration, 'ramp', np.radians(-dlc_angle), 0.0),
    ]

    def get_heading(ti):
        # DLC takes priority if active
        for (t0, t1, kind, v0, v1) in dlc_segments:
            if t0 <= ti < t1:
                return smooth_interp(ti, t0, t1, v0, v1) + get_heading(dlc_start - dt)

        # Otherwise use the randomised heading segments
        for (t0, t1, kind, v0, v1) in heading_segments:
            if t0 <= ti < t1:
                if kind == 'hold':
                    return v0
                else:
                    return smooth_interp(ti, t0, t1, v0, v1)

        return heading_segments[-1][3]  # hold last value after all segments

    heading_profile = np.array([get_heading(times[i]) for i in range(N)])

    # --- Target yaw rate profile ---
    # Derivative of heading profile — used in error metric to penalise
    # yaw rate tracking error, which is directly sensitive to Iz
    target_yaw_rate = np.gradient(heading_profile, dt)

    return {
        "target_speeds":    speed_profile,
        "target_headings":  heading_profile,
        "target_yaw_rates": target_yaw_rate,
    }


def generate_disturbance_profile(N: int, dt: float, rng: np.random.Generator) -> dict:
    wind_amp_long = rng.uniform(0.7, 1)
    wind_amp_lat = rng.uniform(0.7, 1)
    wind_amp_yaw = rng.uniform(0.7, 1)
    road_amp = rng.uniform(0.7, 1)

    wind_freq_long = rng.uniform(0.4, 1.6)
    wind_freq_lat = rng.uniform(0.4, 1.6)
    wind_freq_yaw = rng.uniform(0.4, 1.6)
    road_freq = rng.uniform(0.04, 0.4)

    wind_phase_long = rng.uniform(0.0, 2 * np.pi)
    wind_phase_lat = rng.uniform(0.0, 2 * np.pi)
    wind_phase_yaw = rng.uniform(0.0, 2 * np.pi)
    road_phase = rng.uniform(0.0, 2 * np.pi)

    noise_long = rng.normal(0.0, 0.2, size=N)
    noise_lat = rng.normal(0.0, 0.6, size=N)
    noise_yaw = rng.normal(0.0, np.radians(0.8), size=N)
    noise_road = rng.normal(0.0, np.radians(0.03), size=N)

    wind_longs = np.zeros(N)
    wind_lats = np.zeros(N)
    wind_yaws = np.zeros(N)
    road_grades = np.zeros(N)

    wind_long = wind_lat = wind_yaw = road_grade = 0.0

    for i in range(N):
        t = i * dt

        road_grade += road_amp * noise_road[i] * dt
        road_grade += road_amp * np.radians(1.0) * np.sin(road_freq * t + road_phase) * dt
        road_grade = np.clip(road_grade, np.radians(-10), np.radians(10))

        wind_long += wind_amp_long * noise_long[i] * dt
        wind_long += wind_amp_long * 0.1 * np.sin(wind_freq_long * t + wind_phase_long) * dt
        wind_long = np.clip(wind_long, -0.5, 0.5)

        wind_lat += wind_amp_lat * noise_lat[i] * dt
        wind_lat += wind_amp_lat * 0.2 * np.sin(wind_freq_lat * t + wind_phase_lat) * dt
        wind_lat = np.clip(wind_lat, -0.3, 0.3)

        wind_yaw += wind_amp_yaw * noise_yaw[i] * dt
        wind_yaw += wind_amp_yaw * np.radians(1.5) * np.sin(wind_freq_yaw * t + wind_phase_yaw) * dt
        wind_yaw = np.clip(wind_yaw, -np.radians(8), np.radians(8))

        wind_longs[i] = wind_long
        wind_lats[i] = wind_lat
        wind_yaws[i] = wind_yaw
        road_grades[i] = road_grade

    return {
        "wind_longs": wind_longs,
        "wind_lats": wind_lats,
        "wind_yaws": wind_yaws,
        "road_grades": road_grades,
    }



SHARED_TARGET_PROFILES = [
    generate_reference_profile(N, DT, np.random.default_rng(seed))
    for seed in range(N_PROFILE_RUNS)
]

SHARED_DISTURBANCE_PROFILES = [
    generate_disturbance_profile(N, DT, np.random.default_rng(seed + 500_000))
    for seed in range(N_PROFILE_RUNS)
]

# Fixed baseline values used while optimising the other loop.
# These do not need to be perfect; they just keep the other loop stable.
BASE_STEER_PID = (350, 25, 20)
BASE_SPEED_PID = (40, 900, 0)

STEER_BOUNDS = [
    (0, 1000),  # kp
    (10, 65),   # ki
    (0, 50),   # kd
]

SPEED_BOUNDS = [
    (0, 200),  # kp
    (500, 1500),   # ki
    (0, 2),   # kd
]


# -----------------------------------------------------------------------------
# Car parameter container
# -----------------------------------------------------------------------------

@dataclass
class CarParams:
    mass: float
    max_acceleration: float
    yaw_inertia: float
    length: float
    max_steering_stiffness_front: float
    max_steering_stiffness_rear: float
    tyre_friction: float
    cd: float
    cross_sectional_area: float
    rho_air: float
    max_steer_angle: float
    max_steer_rate: float
    min_accel: float

    def ml_dict(self) -> dict:
        return {
            "mass": self.mass,
            "max_acceleration": self.max_acceleration,
            "yaw_inertia": self.yaw_inertia,
            "length": self.length,
            "max_steering_stiffness_front": self.max_steering_stiffness_front,
            "max_steering_stiffness_rear": self.max_steering_stiffness_rear,
            "tyre_friction": self.tyre_friction,
            "cd": self.cd,
            "cross_sectional_area": self.cross_sectional_area,
        }

# -----------------------------------------------------------------------------
# Car generation
# -----------------------------------------------------------------------------

def generate_random_car_params(rng: np.random.Generator) -> CarParams:
    m = float(round(rng.uniform(800, 4500)))
    L = float(rng.uniform(2.2, 4.8))
    mu = float(rng.uniform(0.3, 1.4))
    Cd = float(rng.uniform(0.18, 0.55))
    A = float(rng.uniform(1.6, 3.8))
    rho_air = 1.225

    max_accel = float(rng.uniform(0.8, 9.0))

    front_weight_frac = float(rng.uniform(0.45, 0.60))
    Lr = front_weight_frac * L
    Lf = L - Lr

    k_Iz = float(rng.uniform(0.20, 0.36))
    Iz = float(round(k_Iz * m * L ** 2))

    g = 9.81
    Fz_front = m * g * (Lr / L)
    Fz_rear = m * g * (Lf / L)

    c_alpha_scale_front = float(rng.uniform(6.0, 12.0))
    c_alpha_scale_rear = float(rng.uniform(6.0, 12.0))

    Cf = float(c_alpha_scale_front * Fz_front)
    Cr = float(c_alpha_scale_rear * Fz_rear)

    base_max_steer_deg = np.clip(35.0 - 4.0 * (L - 2.2) / (3.2 - 2.2), 25.0, 45.0)
    max_steer_angle = float(np.radians(base_max_steer_deg))
    max_steer_rate = float(np.radians(90.0))
    min_accel = float(-mu * 9.81 * rng.uniform(0.6, 0.9))

    return CarParams(
        mass=m,
        max_acceleration=max_accel,
        yaw_inertia=Iz,
        length=L,
        max_steering_stiffness_front=Cf,
        max_steering_stiffness_rear=Cr,
        tyre_friction=mu,
        cd=Cd,
        cross_sectional_area=A,
        rho_air=rho_air,
        max_steer_angle=max_steer_angle,
        max_steer_rate=max_steer_rate,
        min_accel=min_accel,
    )


def make_car(params: CarParams, steer_pid: tuple[float, float, float], speed_pid: tuple[float, float, float]) -> Car:
    return Car(
        m=params.mass,
        Iz=params.yaw_inertia,
        L=params.length,
        Cf=params.max_steering_stiffness_front,
        Cr=params.max_steering_stiffness_rear,
        mu=params.tyre_friction,
        Cd=params.cd,
        A=params.cross_sectional_area,
        rho_air=params.rho_air,
        max_steer_angle=params.max_steer_angle,
        max_steer_rate=params.max_steer_rate,
        max_accel=params.max_acceleration,
        min_accel=params.min_accel,
        kp_steer=steer_pid[0],
        ki_steer=steer_pid[1],
        kd_steer=steer_pid[2],
        kp_speed=speed_pid[0],
        ki_speed=speed_pid[1],
        kd_speed=speed_pid[2],
    )


# -----------------------------------------------------------------------------
# Simulation and objective functions
# -----------------------------------------------------------------------------

def simulate_pid(
    params: CarParams,
    steer_pid: tuple[float, float, float],
    speed_pid: tuple[float, float, float],
    target_profiles: list[dict],
    disturbance_profiles: list[dict],
    plot: bool,
    dt: float = DT,
) -> tuple[float, float]:
    total_speed_error = 0.0
    total_heading_error = 0.0
    n_runs = len(target_profiles)

    for run_idx in range(n_runs):
        car = make_car(params, steer_pid=steer_pid, speed_pid=speed_pid)
        car.reset_pid()

        target_profile = target_profiles[run_idx]
        disturbance_profile = disturbance_profiles[run_idx]

        target_speeds = target_profile["target_speeds"]
        target_headings = target_profile["target_headings"]
        target_yaw_rates = target_profile["target_yaw_rates"]

        wind_longs = disturbance_profile["wind_longs"]
        wind_lats = disturbance_profile["wind_lats"]
        wind_yaws = disturbance_profile["wind_yaws"]
        road_grades = disturbance_profile["road_grades"]
        states = []
        times = np.arange(0, T, dt)
        psis = []

        state = np.array([0.0, 0.0, 0.0, 0, 0.0, 0.0])

        speed_err_accumulative = 0.0
        heading_err_accumulative = 0.0

        prev_accel_cmd = 0.0
        prev_steer_cmd = 0.0

        for i in range(len(target_speeds)):
            target_speed = target_speeds[i]
            target_heading = target_headings[i]

            state, steer_cmd, accel_cmd = car.controlled_step(
                state=state,
                target_heading=target_heading,
                target_speed=target_speed,
                dt=dt,
            )

            x, y, psi, v, vy, r = state

            # Apply disturbances after controller step, matching your existing setup.
            a_grade = -9.81 * np.sin(road_grades[i])
            v += (a_grade + wind_longs[i]) * dt
            vy += wind_lats[i] * dt
            r += wind_yaws[i] * dt
            psi += r * dt
            psi = car.wrap_angle(psi)
            v = max(v, 0.0)

            state = np.array([x, y, psi, v, vy, r])

            speed_err = abs(target_speed - np.linalg.norm([v, vy]))
            heading_err = abs(car.wrap_angle(psi - target_heading))
            yaw_rate_err = abs(r - target_yaw_rates[i])

            # Normalised control effort terms
            accel_effort = (accel_cmd / (car.max_accel + 1e-9)) ** 2
            steer_effort = (steer_cmd / (car.max_steer_angle + 1e-9)) ** 2

            # Saturation penalties
            accel_saturation = 0.0
            steer_saturation = 0.0

            if abs(accel_cmd) >= 0.98 * max(abs(car.max_accel), abs(car.min_accel)):
                accel_saturation = 1.0

            if abs(steer_cmd) >= 0.98 * car.max_steer_angle:
                steer_saturation = 1.0

            # Final objective accumulation
            accel_rate = ((accel_cmd - prev_accel_cmd) / (car.max_accel + 1e-9)) ** 2
            steer_rate = ((steer_cmd - prev_steer_cmd) / (car.max_steer_angle + 1e-9)) ** 2

            speed_err_accumulative += (
                speed_err
                + 0.2 * accel_effort
                + 0.001 * accel_rate
                + 2.0 * accel_saturation
            ) * dt

            heading_err_accumulative += (
                8.0 * heading_err
                + 5.0 * yaw_rate_err
                + 0.5 * steer_effort
                + 0.001 * steer_rate
                + 3.0 * steer_saturation
            ) * dt

            states.append(state)
            psis.append(psi)

            prev_accel_cmd = accel_cmd
            prev_steer_cmd = steer_cmd

        total_speed_error += speed_err_accumulative / n_runs
        total_heading_error += heading_err_accumulative / n_runs

    if plot==True:
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

    return total_speed_error, total_heading_error


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

def make_shared_profiles() -> tuple[list[dict], list[dict]]:
    """
    Generate one deterministic profile bank shared by every car.

    This matches the first file's structure:
    every car is tested against the same target and disturbance profiles.
    """
    target_profiles = [
        generate_reference_profile(N, DT, np.random.default_rng(seed))
        for seed in range(N_PROFILE_RUNS)
    ]

    disturbance_profiles = [
        generate_disturbance_profile(N, DT, np.random.default_rng(seed + 500_000))
        for seed in range(N_PROFILE_RUNS)
    ]

    return target_profiles, disturbance_profiles


def optimise_speed_pid(
    params: CarParams,
    target_profiles: list[dict],
    disturbance_profiles: list[dict],
    fixed_steer_pid: tuple[float, float, float] = BASE_STEER_PID,
) -> tuple[tuple[float, float, float], float]:
    def objective(speed_pid_array: np.ndarray) -> float:
        speed_pid = tuple(float(v) for v in speed_pid_array)
        speed_error, _ = simulate_pid(
            params=params,
            steer_pid=fixed_steer_pid,
            speed_pid=speed_pid,
            target_profiles=target_profiles,
            disturbance_profiles=disturbance_profiles,
            plot=False,
        )
        return speed_error

    res = differential_evolution(
        objective,
        bounds=SPEED_BOUNDS,
        seed=42,
        maxiter=25,
        popsize=8,
        polish=False,
        workers=1,
    )

    # Cheap local polish from the DE result.
    local = minimize(
        objective,
        x0=res.x,
        method="L-BFGS-B",
        bounds=SPEED_BOUNDS,
        options={"maxiter": 80},
    )

    best = local.x if local.fun <= res.fun else res.x
    best_error = float(min(local.fun, res.fun))
    return tuple(float(v) for v in best), best_error


def optimise_steer_pid(
    params: CarParams,
    target_profiles: list[dict],
    disturbance_profiles: list[dict],
    fixed_speed_pid: tuple[float, float, float] = BASE_SPEED_PID,
) -> tuple[tuple[float, float, float], float]:
    def objective(steer_pid_array: np.ndarray) -> float:
        steer_pid = tuple(float(v) for v in steer_pid_array)
        _, heading_error = simulate_pid(
            params=params,
            steer_pid=steer_pid,
            speed_pid=fixed_speed_pid,
            target_profiles=target_profiles,
            disturbance_profiles=disturbance_profiles,
            plot=False
        )
        return heading_error

    res = differential_evolution(
        objective,
        bounds=STEER_BOUNDS,
        seed=43,
        maxiter=25,
        popsize=8,
        polish=False,
        workers=1,
    )

    local = minimize(
        objective,
        x0=res.x,
        method="L-BFGS-B",
        bounds=STEER_BOUNDS,
        options={"maxiter": 80},
    )

    best = local.x if local.fun <= res.fun else res.x
    best_error = float(min(local.fun, res.fun))
    return tuple(float(v) for v in best), best_error


def optimise_car_pid(
    params: CarParams,
    target_profiles: list[dict],
    disturbance_profiles: list[dict],
) -> dict:
    """
    Optimise speed and steering PID separately.

    The second pass lets the speed optimiser use the newly found steering PID,
    and the steering optimiser use the newly found speed PID.
    """
    speed_pid, _ = optimise_speed_pid(
        params=params,
        target_profiles=target_profiles,
        disturbance_profiles=disturbance_profiles,
        fixed_steer_pid=BASE_STEER_PID,
    )

    steer_pid, _ = optimise_steer_pid(
        params=params,
        target_profiles=target_profiles,
        disturbance_profiles=disturbance_profiles,
        fixed_speed_pid=speed_pid,
    )

    speed_pid, _ = optimise_speed_pid(
        params=params,
        target_profiles=target_profiles,
        disturbance_profiles=disturbance_profiles,
        fixed_steer_pid=steer_pid,
    )

    final_speed_error, final_heading_error = simulate_pid(
        params=params,
        steer_pid=steer_pid,
        speed_pid=speed_pid,
        target_profiles=target_profiles,
        disturbance_profiles=disturbance_profiles,
        plot=False,
    )

    return {
        "opt_steer_kp": steer_pid[0],
        "opt_steer_ki": steer_pid[1],
        "opt_steer_kd": steer_pid[2],
        "opt_speed_kp": speed_pid[0],
        "opt_speed_ki": speed_pid[1],
        "opt_speed_kd": speed_pid[2],
        "best_speed_error": final_speed_error,
        "best_heading_error": final_heading_error,
    }


# -----------------------------------------------------------------------------
# Dataset generation
# -----------------------------------------------------------------------------

def write_header(writer: csv.writer) -> None:
    writer.writerow(
        CAR_PARAM_COLS
        + [
            "opt_steer_kp",
            "opt_steer_ki",
            "opt_steer_kd",
            "opt_speed_kp",
            "opt_speed_ki",
            "opt_speed_kd",
            "best_speed_error",
            "best_heading_error",
        ]
    )


def generate_optimal_pid_dataset(output_csv: str = OUTPUT_CSV, num_cars: int = NUM_CARS) -> None:
    target_profiles, disturbance_profiles = make_shared_profiles()

    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        write_header(writer)

        for car_idx in range(num_cars):
            params = generate_random_car_params(RNG)

            print("\n" + "=" * 70)
            print(f"Optimising car {car_idx + 1}/{num_cars}")
            print(params.ml_dict())

            result = optimise_car_pid(
                params=params,
                target_profiles=target_profiles,
                disturbance_profiles=disturbance_profiles,
            )

            row_dict = {**params.ml_dict(), **result}
            writer.writerow([row_dict[col] for col in CAR_PARAM_COLS] + [
                row_dict["opt_steer_kp"],
                row_dict["opt_steer_ki"],
                row_dict["opt_steer_kd"],
                row_dict["opt_speed_kp"],
                row_dict["opt_speed_ki"],
                row_dict["opt_speed_kd"],
                row_dict["best_speed_error"],
                row_dict["best_heading_error"],
            ])

            print(
                "Best PID: "
                f"steer=({result['opt_steer_kp']:.3f}, {result['opt_steer_ki']:.3f}, {result['opt_steer_kd']:.3f}), "
                f"speed=({result['opt_speed_kp']:.3f}, {result['opt_speed_ki']:.3f}, {result['opt_speed_kd']:.3f})"
            )
            print(
                f"Errors: speed={result['best_speed_error']:.3f}, "
                f"heading={result['best_heading_error']:.3f}"
            )

    print(f"\nSaved optimal PID dataset -> {output_csv}")


if __name__ == "__main__":
    generate_optimal_pid_dataset(
        output_csv=OUTPUT_CSV,
        num_cars=NUM_CARS,
    )

    train_direct_pid_models(
        csv_path=OUTPUT_CSV,
        steer_model_path="direct_steer_pid_model.pkl",
        speed_model_path="direct_speed_pid_model.pkl",
    )

    plt.show()
