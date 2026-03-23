import numpy as np
import matplotlib.pyplot as plt
from Car import Car

num_simulations = 20

cars = []

rng = np.random.default_rng()

for _ in range(num_simulations):

        # -------------------------------------------------
    # 1. Primary sampled parameters
    # -------------------------------------------------
    m = rng.uniform(1000.0, 2200.0)          # kg
    L = rng.uniform(2.4, 3.2)                # m
    mu = rng.uniform(0.6, 1.2)               # road friction
    Cd = rng.uniform(0.24, 0.40)             # realistic passenger-car range
    A = rng.uniform(1.8, 2.8)                # frontal area [m^2]
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
    Iz = k_Iz * m * L**2

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

    # Small extra uncertainty
    Cf *= rng.uniform(0.9, 1.1)
    Cr *= rng.uniform(0.9, 1.1)

    # -------------------------------------------------
    # 4. Steering limits
    # -------------------------------------------------
    # Smaller wheelbase cars tend to steer a bit more sharply
    base_max_steer_deg = 35.0 - 4.0 * (L - 2.4) / (3.2 - 2.4)
    max_steer_angle = np.radians(base_max_steer_deg * rng.uniform(0.9, 1.1))

    # Steering rate also varies, but keep it realistic
    max_steer_rate = np.radians(rng.uniform(60.0, 120.0))

    # -------------------------------------------------
    # 5. Longitudinal capability
    # -------------------------------------------------
    # Use acceleration capability ranges typical of passenger vehicles
    max_accel = rng.uniform(2.0, 4.0)        # m/s^2
    min_accel = -rng.uniform(4.0, 8.0)       # braking, negative

    # -------------------------------------------------
    # 6. PID gains
    # -------------------------------------------------
    # Still random for now, but in a tighter, more realistic band
    kp_steer = rng.uniform(1.5, 3.5)
    ki_steer = rng.uniform(0.02, 0.15)
    kd_steer = rng.uniform(0.2, 0.8)

    kp_speed = rng.uniform(0.6, 1.4)
    ki_speed = rng.uniform(0.05, 0.25)
    kd_speed = rng.uniform(0.02, 0.15)

    # ---------------------------
    # Create car
    # ---------------------------
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
        kp_steer=kp_steer,
        ki_steer=ki_steer,
        kd_steer=kd_steer,
        kp_speed=kp_speed,
        ki_speed=ki_speed,
        kd_speed=kd_speed
    )

    cars.append(car)

def print_compact(car, err):
    print(
        f"{car.m:.1f}, {car.Iz:.1f}, {car.L:.2f}, "
        f"{car.Cf:.0f}, {car.Cr:.0f}, {car.mu:.2f}, "
        f"{car.Cd:.2f}, {car.A:.2f}, "
        f"{car.kp_steer:.2f}, {car.ki_steer:.2f}, {car.kd_steer:.2f}, "
        f"{car.kp_speed:.2f}, {car.ki_speed:.2f}, {car.kd_speed:.2f}, "
        f"{err:.2f}"
    )

def plot_drive_conditions_stacked(times, states, target_speed, wind_longs, wind_lats, road_grades):
    """
    Plot:
    1) Speed vs target speed
    2) Wind direction
    3) Road gradient

    All sharing the same time axis.
    """

    # ---------------------------
    # Compute signals
    # ---------------------------
    speed = np.linalg.norm(states[:, 3:5], axis=1)

    if np.isscalar(target_speed):
        target_speed_arr = np.full_like(times, target_speed, dtype=float)
    else:
        target_speed_arr = np.asarray(target_speed)

    wind_direction = np.degrees(np.arctan2(wind_lats, wind_longs))
    road_gradient_deg = np.degrees(road_grades)

    # ---------------------------
    # Create figure
    # ---------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(14, 10),
        sharex=True
    )

    # ---------------------------
    # 1. Speed plot
    # ---------------------------
    ax1.plot(times, speed, label="Actual Speed [m/s]", linewidth=2)
    ax1.plot(times, target_speed_arr, linestyle="--", label="Target Speed [m/s]", linewidth=2)
    ax1.set_ylim([0, 40])
    ax1.set_ylabel("Speed [m/s]")
    ax1.set_title("Vehicle Performance and Disturbances")
    ax1.legend()
    ax1.grid(True)

    # ---------------------------
    # 2. Wind direction
    # ---------------------------
    ax2.plot(times, wind_direction, linestyle=":", linewidth=2)
    ax2.set_ylabel("Wind Direction [deg]")
    ax2.grid(True)

    # ---------------------------
    # 3. Road gradient
    # ---------------------------
    ax3.plot(times, road_gradient_deg, linestyle="-.", linewidth=2)
    ax3.set_ylabel("Road Gradient [deg]")
    ax3.set_xlabel("Time [s]")
    ax3.grid(True)

    # ---------------------------
    # Layout
    # ---------------------------
    plt.tight_layout()


dt = 0.01
T = 90.0
N = int(T / dt)
print("m, Iz, L, Cf, Cr, mu, Cd, A, kp_steer, ki_steer, kd_steer, kp_speed, ki_speed, kd_speed, err")

for car in cars:
    state = np.array([0.0, 0.0, 0.0, 30, 0.0, 0.0])

    states = []
    steers = []
    accels = []
    times = []

    # Retained disturbance histories
    target_headings = []
    road_grades = []
    wind_longs = []
    wind_lats = []
    wind_yaws = []

    err_accumulative = 0.0

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

    for i in range(N):
        t = i * dt

        # Lane heading
        lane_heading += lane_amp * rng.normal(0.0, np.radians(0.15)) * dt
        lane_heading += lane_amp * np.radians(5.0) * np.sin(lane_freq * t + lane_phase) * dt
        lane_heading = np.clip(lane_heading, np.radians(-45), np.radians(45))
        target_heading = lane_heading

        # Road gradient
        road_grade += road_amp * rng.normal(0.0, np.radians(0.03)) * dt
        road_grade += road_amp * np.radians(1.0) * np.sin(road_freq * t + road_phase) * dt
        road_grade = np.clip(road_grade, np.radians(-10), np.radians(10))

        # Wind
        wind_long += wind_amp_long * rng.normal(0.0, 0.03) * dt
        wind_lat  += wind_amp_lat  * rng.normal(0.0, 0.4) * dt
        wind_yaw  += wind_amp_yaw  * rng.normal(0.0, np.radians(0.8)) * dt

        wind_long += wind_amp_long * 0.3 * np.sin(wind_freq_long * t + wind_phase_long) * dt
        wind_lat  += wind_amp_lat  * 0.5 * np.sin(wind_freq_lat * t + wind_phase_lat) * dt
        wind_yaw  += wind_amp_yaw  * np.radians(1.5) * np.sin(wind_freq_yaw * t + wind_phase_yaw) * dt

        wind_long = np.clip(wind_long, -0.5, 0.5)
        wind_lat  = np.clip(wind_lat, -0.3, 0.3)
        wind_yaw  = np.clip(wind_yaw, -np.radians(8), np.radians(8))

        # Base desired speed
        target_speed = 30.0

        # =========================================================
        # 3. Controlled vehicle step
        # =========================================================
        state, steer_cmd, accel_cmd = car.controlled_step(
            state=state,
            target_heading=target_heading,
            target_speed=target_speed,
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
        v  += (a_grade + wind_long) * dt
        vy += wind_lat * dt
        r  += wind_yaw * dt

        # Update heading after yaw disturbance
        psi += r * dt
        psi = car.wrap_angle(psi)

        # Prevent negative speed if desired
        v = max(v, 0.0)

        state = np.array([x, y, psi, v, vy, r])

        # =========================================================
        # 6. Performance metric
        # =========================================================
        speed_mag = np.linalg.norm([state[3], state[4]])
        err = target_speed - speed_mag
        err_accumulative += abs(err) * dt

        # =========================================================
        # 7. Store results
        # =========================================================
        states.append(state.copy())
        steers.append(steer_cmd)
        accels.append(accel_cmd)
        times.append(t)

        # Store disturbance history
        target_headings.append(target_heading)
        road_grades.append(road_grade)
        wind_longs.append(wind_long)
        wind_lats.append(wind_lat)
        wind_yaws.append(wind_yaw)

    states = np.array(states)
    steers = np.array(steers)
    accels = np.array(accels)
    times = np.array(times)

    target_headings = np.array(target_headings)
    road_grades = np.array(road_grades)
    wind_longs = np.array(wind_longs)
    wind_lats = np.array(wind_lats)
    wind_yaws = np.array(wind_yaws)

    print_compact(car, err_accumulative)

    if err_accumulative < 15 or err_accumulative > 30:
        plot_drive_conditions_stacked(
        times=times,
        states=states,
        target_speed=30.0,
        wind_longs=wind_longs,
        wind_lats=wind_lats,
        road_grades=road_grades
    )

plt.show()
