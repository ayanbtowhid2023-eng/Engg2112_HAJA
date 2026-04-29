import numpy as np
import matplotlib.pyplot as plt
from Car import Car
import csv

# csv_file = open('DM_results_2.csv', 'a', newline='')
# writer = csv.writer(csv_file)

# # Header row
# writer.writerow([
#     "mass",
#     "max_acceleration",
#     "yaw_inertia",
#     "length",
#     "max_steering_stiffness_front",
#     "max_steering_stiffness_rear",
#     "tyre_friction",
#     "cd",
#     "cross_sectional_area",
#     "steer_kp", "steer_ki", "steer_kd",
#     "speed_kp",  "speed_ki",  "speed_kd",
#     "simulation_error"
# ])

num_simulations = 1

cars = []

rng = np.random.default_rng()
    
def generate_reference_profile(N, dt, rng):
    """
    Generate a seeded target speed and heading profile.
    Structure: alternating transition and hold segments.
    """
    T = N * dt
    times = np.arange(N) * dt

    # --- Speed profile ---
    # Randomise the speed levels and transition times but keep hold sections
    speeds = [
        rng.uniform(15.0, 25.0),   # initial hold speed
        rng.uniform(25.0, 35.0),   # second level
        rng.uniform(10.0, 20.0),   # drop
        rng.uniform(20.0, 30.0),   # final level
    ]

    # Transition durations and hold durations
    hold_dur   = [rng.uniform(12.0, 20.0) for _ in range(4)]  # hold each level
    trans_dur  = [rng.uniform(2.0,  6.0)  for _ in range(3)]  # ramp between levels

    speed_profile = np.zeros(N)
    t_cursor = 0.0

    def smooth_interp(t, t0, t1, v0, v1):
        s = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
        return v0 + (v1 - v0) * 0.5 * (1 - np.cos(np.pi * s))

    current_speed = speeds[0]
    segments = []

    # Build segment list: (start_t, end_t, type, from_val, to_val)
    t = 0.0
    for i in range(len(speeds)):
        segments.append((t, t + hold_dur[i], 'hold', speeds[i], speeds[i]))
        t += hold_dur[i]
        if i < len(speeds) - 1:
            segments.append((t, t + trans_dur[i], 'ramp', speeds[i], speeds[i+1]))
            t += trans_dur[i]

    for i in range(N):
        ti = times[i]
        val = speeds[0]
        for (t0, t1, kind, v0, v1) in segments:
            if t0 <= ti < t1:
                if kind == 'hold':
                    val = v0
                else:
                    val = smooth_interp(ti, t0, t1, v0, v1)
                break
            val = segments[-1][3]  # after all segments, hold last value
        speed_profile[i] = val

    # --- Heading profile ---
    max_angle = rng.uniform(15.0, 30.0)   # peak turn angle in degrees

    # Randomise turn angles (signed) and timing
    angles_deg = [0.0]
    for _ in range(4):
        sign = rng.choice([-1, 1])
        angles_deg.append(float(sign * rng.uniform(10.0, max_angle)))
    angles_deg.append(0.0)  # always return to zero at end

    h_hold_dur  = [rng.uniform(8.0,  16.0) for _ in range(len(angles_deg))]
    h_trans_dur = [rng.uniform(4.0,  8.0)  for _ in range(len(angles_deg) - 1)]

    heading_segments = []
    t = 0.0
    for i in range(len(angles_deg)):
        heading_segments.append((t, t + h_hold_dur[i], 'hold',
                                  np.radians(angles_deg[i]),
                                  np.radians(angles_deg[i])))
        t += h_hold_dur[i]
        if i < len(angles_deg) - 1:
            heading_segments.append((t, t + h_trans_dur[i], 'ramp',
                                      np.radians(angles_deg[i]),
                                      np.radians(angles_deg[i+1])))
            t += h_trans_dur[i]

    heading_profile = np.zeros(N)
    for i in range(N):
        ti = times[i]
        val = np.radians(angles_deg[0])
        for (t0, t1, kind, v0, v1) in heading_segments:
            if t0 <= ti < t1:
                if kind == 'hold':
                    val = v0
                else:
                    s = np.clip((ti - t0) / (t1 - t0), 0.0, 1.0)
                    val = v0 + (v1 - v0) * 0.5 * (1 - np.cos(np.pi * s))
                break
            val = heading_segments[-1][3]
        heading_profile[i] = val

    return {
        "target_speeds":   speed_profile,
        "target_headings": heading_profile,
    }

def generate_disturbance_profile(N, dt, rng):
    """Pre-generate a full disturbance sequence so it can be replayed."""

    wind_amp_long = rng.uniform(0.4, 0.6)
    wind_amp_lat  = rng.uniform(0.4, 0.6)
    wind_amp_yaw  = rng.uniform(0.4, 0.6)
    road_amp      = rng.uniform(0.4, 0.6)

    wind_freq_long = rng.uniform(0.4, 1.6)
    wind_freq_lat  = rng.uniform(0.4, 1.6)
    wind_freq_yaw  = rng.uniform(0.4, 1.6)
    road_freq      = rng.uniform(0.04, 0.4)

    wind_phase_long = rng.uniform(0.0, 2 * np.pi)
    wind_phase_lat  = rng.uniform(0.0, 2 * np.pi)
    wind_phase_yaw  = rng.uniform(0.0, 2 * np.pi)
    road_phase      = rng.uniform(0.0, 2 * np.pi)

    # Pre-draw all random noise samples for the entire run
    noise_long = rng.normal(0.0, 0.2,            size=N)
    noise_lat  = rng.normal(0.0, 0.6,            size=N)
    noise_yaw  = rng.normal(0.0, np.radians(0.8), size=N)
    noise_road = rng.normal(0.0, np.radians(0.03), size=N)

    wind_longs     = np.zeros(N)
    wind_lats      = np.zeros(N)
    wind_yaws      = np.zeros(N)
    road_grades    = np.zeros(N)

    wind_long = wind_lat = wind_yaw = road_grade = 0.0

    for i in range(N):
        t = i * dt

        road_grade += road_amp * noise_road[i] * dt
        road_grade += road_amp * np.radians(1.0) * np.sin(road_freq * t + road_phase) * dt
        road_grade  = np.clip(road_grade, np.radians(-10), np.radians(10))

        wind_long += wind_amp_long * noise_long[i] * dt
        wind_long += wind_amp_long * 0.1 * np.sin(wind_freq_long * t + wind_phase_long) * dt
        wind_long  = np.clip(wind_long, -0.5, 0.5)

        wind_lat  += wind_amp_lat * noise_lat[i] * dt
        wind_lat  += wind_amp_lat * 0.2 * np.sin(wind_freq_lat * t + wind_phase_lat) * dt
        wind_lat   = np.clip(wind_lat, -0.3, 0.3)

        wind_yaw  += wind_amp_yaw * noise_yaw[i] * dt
        wind_yaw  += wind_amp_yaw * np.radians(1.5) * np.sin(wind_freq_yaw * t + wind_phase_yaw) * dt
        wind_yaw   = np.clip(wind_yaw, -np.radians(8), np.radians(8))

        wind_longs[i]    = wind_long
        wind_lats[i]     = wind_lat
        wind_yaws[i]     = wind_yaw
        road_grades[i]   = road_grade

    return {
        "wind_longs":    wind_longs,
        "wind_lats":     wind_lats,
        "wind_yaws":     wind_yaws,
        "road_grades":   road_grades,
    }

for _ in range(num_simulations):

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
    base_max_steer_deg = 35.0 - 4.0 * (L - 2.2) / (3.2 - 2.2)
    max_steer_angle = np.radians(base_max_steer_deg)

    # Steering rate also varies, but keep it realistic
    max_steer_rate = np.radians(90)

    # -------------------------------------------------
    # 5. Longitudinal capability
    # -------------------------------------------------
    # Use acceleration capability ranges typical of passenger vehicles
    max_accel = rng.uniform(0.8, 9.0) 
    min_accel = -6       # braking, negative

    # -------------------------------------------------
    # 6. PID gains
    # -------------------------------------------------
    kp_steer = rng.uniform(0, 3)
    ki_steer = rng.uniform(0, 1.5)
    kd_steer = rng.uniform(0, 1)

    kp_speed = rng.uniform(0, 28)
    ki_speed = rng.uniform(0, 6)
    kd_speed = rng.uniform(0, 1)

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
        f"{car.m:.0f}, {car.max_accel:.2f}, {car.Iz:.0f}, {car.L:.2f}, "
        f"{car.Cf:.0f}, {car.Cr:.0f}, {car.mu:.2f}, "
        f"{car.Cd:.2f}, {car.A:.2f}, "
        f"{car.kp_steer:.2f}, {car.ki_steer:.2f}, {car.kd_steer:.2f}, "
        f"{car.kp_speed:.2f}, {car.ki_speed:.2f}, {car.kd_speed:.2f}, "
        f"      {err:.2f}"
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


dt = 0.01
T = 90.0
N = int(T / dt)
N2 = 10 # Number of runs to normalise error distribution 
# print(" m    max_a  Iz    L      Cf     Cr    mu    Cd     A    p_s   i_s   d_s   p_v    i_v   d_v         err")

disturbance_profiles = [
    generate_disturbance_profile(N, dt, np.random.default_rng(seed))
    for seed in range(N2)
]

target_profile = [
    generate_reference_profile(N, dt, np.random.default_rng(seed))
    for seed in range(N2)
]

for car in cars:

    avg_error = 0

    for _ in range(N2):
        car.reset_pid()

        states = []
        steers = []
        accels = []
        times = []
        psis = []

        t = 0

        # Retained disturbance histories

        profile = disturbance_profiles[_]
        profile_targets = target_profile[_]

        err_accumulative = 0.0

        wind_longs = profile["wind_longs"]
        wind_lats = profile["wind_lats"]
        wind_yaws = profile["wind_yaws"]
        road_grades = profile["road_grades"]
        target_speeds = profile_targets["target_speeds"]
        target_headings = profile_targets["target_headings"]

        state = np.array([0.0, 0.0, 0.0, target_speeds[0], 0.0, 0.0])

        for i in range(N):
            wind_long      = wind_longs[i]
            wind_lat       = wind_lats[i]
            wind_yaw       = wind_yaws[i]
            road_grade     = road_grades[i]
            target_speed   = target_speeds[i]
            target_heading = target_headings[i]


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
            psi = car.wrap_angle(psi)

            # Prevent negative speed if desired
            v = max(v, 0.0)

            state = np.array([x, y, psi, v, vy, r])

            # =========================================================
            # 6. Performance metric
            # =========================================================
            speed_err   = abs(target_speed - np.linalg.norm([v, vy]))
            heading_err = abs(car.wrap_angle(psi - target_heading))

            err_accumulative += (speed_err + 10.0 * heading_err) * dt

            # =========================================================
            # 7. Store results
            # =========================================================
            states.append(state.copy())
            steers.append(steer_cmd)
            accels.append(accel_cmd)
            times.append(t)
            psis.append(psi)

            t += dt

        states = np.array(states)
        steers = np.array(steers)
        accels = np.array(accels)
        times = np.array(times)
        psis = np.array(psis)

        road_grades = np.array(road_grades)
        wind_longs = np.array(wind_longs)
        wind_lats = np.array(wind_lats)
        wind_yaws = np.array(wind_yaws)
        target_headings = np.array(target_headings)
        target_speeds = np.array(target_speeds)

        avg_error += err_accumulative / N2

    # print_compact(car, err_accumulative)

    # writer.writerow([
    #     f"{car.m:.2f}",
    #     f"{car.max_accel:.2f}",
    #     f"{car.Iz:.2f}",
    #     f"{car.L:.2f}",
    #     f"{car.Cf:.2f}",
    #     f"{car.Cr:.2f}",
    #     f"{car.mu:.2f}",
    #     f"{car.Cd:.2f}",
    #     f"{car.A:.2f}",
    #     f"{car.kp_steer:.2f}",
    #     f"{car.ki_steer:.2f}",
    #     f"{car.kd_steer:.2f}",
    #     f"{car.kp_speed:.2f}",
    #     f"{car.ki_speed:.2f}",
    #     f"{car.kd_speed:.2f}",
    #     f"{avg_error:.2f}"
    # ])

        if avg_error > 0:
            plot_drive_conditions_stacked(
            times=times,
            states=states,
            target_speeds=target_speeds,
            wind_longs=wind_longs,
            wind_lats=wind_lats,
            road_grades=road_grades,
            target_headings = target_headings,
            psis = psis
        )

# csv_file.close()
plt.show()
