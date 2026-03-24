import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from tqdm import tqdm 

# --- Robot parameters ---
wheel_base = 0.18      # distance between wheels (m)
sensor_distance = 0.15 # distance of sensor from robot center (m)
max_speed = 1.5        # max wheel speed (m/s)
base_speed = 1.0       # forward speed (m/s)

# --- Simulation settings ---
dt = 0.01   # timestep
T = 100      # total simulation time (s)
steps = int(T / dt)

# --- Figures ---
fig = plt.figure(figsize=(12, 6))
ax_path = fig.add_axes([0.05, 0.35, 0.4, 0.6])
robot_line, = ax_path.plot([], [], label="Robot Path")
target_line_plot, = ax_path.plot([], [], '--', label="Target Line")
ax_path.set_xlabel("X (m)")
ax_path.set_ylabel("Y (m)")
ax_path.set_title("2D Robot Driving Simulation")
ax_path.legend()

ax_heat = fig.add_axes([0.55, 0.35, 0.4, 0.6])
heatmap_img = ax_heat.imshow(np.zeros((20, 20)), origin='lower', aspect='auto', cmap='hot',
                             extent=[0.5, 3.0, 0.1, 3.0])
cb = fig.colorbar(heatmap_img, ax=ax_heat)
ax_heat.set_xlabel("Period (m)")
ax_heat.set_ylabel("Amplitude (m)")
ax_heat.set_title("Std deviation of error")

# --- Sliders ---
axcolor = 'lightgoldenrodyellow'
ax_kp = fig.add_axes([0.05, 0.25, 0.4, 0.03], facecolor=axcolor)
ax_ki = fig.add_axes([0.05, 0.20, 0.4, 0.03], facecolor=axcolor)
ax_kd = fig.add_axes([0.05, 0.15, 0.4, 0.03], facecolor=axcolor)
ax_amp = fig.add_axes([0.05, 0.10, 0.4, 0.03], facecolor=axcolor)
ax_period = fig.add_axes([0.05, 0.05, 0.4, 0.03], facecolor=axcolor)
ax_offset = fig.add_axes([0.05, 0.00, 0.4, 0.03], facecolor=axcolor)

s_kp = Slider(ax_kp, 'Kp', 0.0, 8.0, valinit=6.85)
s_ki = Slider(ax_ki, 'Ki', 0.0, 4.0, valinit=0.0)
s_kd = Slider(ax_kd, 'Kd', 0.0, 2.0, valinit=0.050)
s_amp = Slider(ax_amp, 'Amplitude (m)', 0.0, 3.0, valinit=0.5)
s_period = Slider(ax_period, 'Period (m)', 0.5, 3.0, valinit=2.0)
s_offset = Slider(ax_offset, 'Offset (m)', -1.0, 1.0, valinit=0.0)

# --- Buttons ---
ax_button = fig.add_axes([0.55, 0.25, 0.2, 0.05])
b_update = Button(ax_button, 'Update Heatmap')

ax_auto = fig.add_axes([0.55, 0.18, 0.2, 0.05])
b_auto = Button(ax_auto, 'Auto-Calibrate')

# --- Target line function ---
def target_line(A, period, offset, x_max=15.0, straight_length=None):
    if straight_length is None:
        straight_length = period
    x_line = np.linspace(0, x_max, int(x_max/dt))
    y_line = np.where(x_line < straight_length,
                      offset,
                      A * np.sin(2 * np.pi * (x_line - straight_length) / period) + offset)
    return x_line, y_line

def angle_diff(a, b):
    diff = a - b
    while diff > np.pi:
        diff -= 2*np.pi
    while diff < -np.pi:
        diff += 2*np.pi
    return diff

# --- True 2D simulation with angular PID ---
def simulate(Kp, Ki, Kd, A, period, offset):
    x, y, theta = 0.0, 0.0, 0.0
    prev_derivative = 0.0
    integral = 0.0
    prev_error = 0.0
    xs, ys = [], []

    x_line, y_line = target_line(A, period, offset, x_max=15.0, straight_length=period)

    for _ in range(steps):
        # Sensor position
        sx = x + sensor_distance * np.cos(theta)
        sy = y + sensor_distance * np.sin(theta)

        # Closest point on line
        distances = (x_line - sx)**2 + (y_line - sy)**2
        idx = np.argmin(distances)
        closest_x, closest_y = x_line[idx], y_line[idx]

        # Vector from robot heading to closest point
        vec_to_line = np.array([closest_x - sx, closest_y - sy])
        heading_vec = np.array([np.cos(theta), np.sin(theta)])

        # Angle between heading and vector to line (signed)
        angle_error = np.arctan2(vec_to_line[0]*heading_vec[1] - vec_to_line[1]*heading_vec[0],
                                 np.dot(heading_vec, vec_to_line))

        # Clip large angular errors to simulate sensor range
        max_angle_error = np.pi / 2  # 90 degrees
        angle_error = np.clip(angle_error, -max_angle_error, max_angle_error)

        # PID controller for angular velocity
        integral += angle_error * dt
        derivative = angle_diff(angle_error, prev_error) / dt
        alpha = 0.1  # small smoothing factor
        derivative = alpha * derivative + (1 - alpha) * prev_derivative
        prev_derivative = derivative
        prev_error = angle_error
        omega = Kp * angle_error + Ki * integral + Kd * derivative

        # Differential drive
        v_l = np.clip(base_speed + (wheel_base / 2) * omega, -max_speed, max_speed)
        v_r = np.clip(base_speed - (wheel_base / 2) * omega, -max_speed, max_speed)
        v_robot = (v_l + v_r) / 2
        omega_robot = (v_r - v_l) / wheel_base

        # Update robot pose
        x += v_robot * np.cos(theta) * dt
        y += v_robot * np.sin(theta) * dt
        theta += omega_robot * dt

        xs.append(x)
        ys.append(y)

        if x >= 15.0:
            break

    return np.array(xs), np.array(ys)

# --- Update plots ---
def update_path(val):
    xs, ys = simulate(s_kp.val, s_ki.val, s_kd.val, s_amp.val, s_period.val, s_offset.val)
    x_line, y_line = target_line(s_amp.val, s_period.val, s_offset.val, x_max=15.0, straight_length=s_period.val)
    robot_line.set_data(xs, ys)
    target_line_plot.set_data(x_line, y_line)
    ax_path.relim()
    ax_path.autoscale_view()
    fig.canvas.draw_idle()

# --- Heatmap ---
def update_heatmap(event):
    amp_vals = np.linspace(0.1, 3.0, 20)
    period_vals = np.linspace(0.5, 3.0, 20)
    heat = np.zeros((len(amp_vals), len(period_vals)))
    for i, A in enumerate(amp_vals):
        for j, p in enumerate(period_vals):
            xs_h, ys_h = simulate(s_kp.val, s_ki.val, s_kd.val, A, p, s_offset.val)
            x_line, y_line = target_line(A, p, s_offset.val, x_max=15.0, straight_length=p)
            y_interp = np.interp(xs_h, x_line, y_line)
            heat[i, j] = np.std(ys_h - y_interp)
    heatmap_img.set_data(heat)
    heatmap_img.set_clim(np.min(heat), np.max(heat))
    ax_heat.set_xlim(0.5, 3.0)
    ax_heat.set_ylim(0.1, 3.0)
    fig.canvas.draw_idle()


# --- Auto-calibration ---
def auto_calibrate(event):
    print("Starting auto-calibration...")
    kp_values = np.arange(0.0, 10.05, 0.05)
    kd_values = np.arange(0.0, 10.05, 0.05)
    Ki_val = 0.0

    # --- Prepare live plot ---
    heat_live = np.zeros((len(kp_values), len(kd_values)))
    heatmap_img.set_data(heat_live)
    heatmap_img.set_clim(0, 1)  # temporary scaling

    # --- Tune Kp ---
    avg_score_kp = []
    for i, kp in enumerate(tqdm(kp_values, desc="Tuning Kp")):
        scores = []
        for A in np.linspace(0.1, 3.0, 10):
            for p in np.linspace(0.5, 3.0, 10):
                xs_h, ys_h = simulate(kp, Ki_val, 0.0, A, p, s_offset.val)
                x_line, y_line = target_line(A, p, s_offset.val, x_max=15.0, straight_length=p)
                y_interp = np.interp(xs_h, x_line, y_line)
                std_dev = np.std(ys_h - y_interp)
                final_x = xs_h[-1] if len(xs_h) > 0 else 0.01
                scores.append(std_dev / final_x)

        avg = np.mean(scores)
        avg_score_kp.append(avg)

        # --- UPDATE PLOT LIVE ---
        heat_live[i, 0] = avg  # fill first column
        heatmap_img.set_data(heat_live)
        heatmap_img.set_clim(np.min(heat_live), np.max(heat_live) + 1e-6)
        fig.canvas.draw_idle()
        plt.pause(0.001)

    best_kp = kp_values[np.argmin(avg_score_kp)]

    # --- Tune Kd ---
    avg_score_kd = []
    for j, kd in enumerate(tqdm(kd_values, desc="Tuning Kd")):
        scores = []
        for A in np.linspace(0.1, 3.0, 10):
            for p in np.linspace(0.5, 3.0, 10):
                xs_h, ys_h = simulate(best_kp, Ki_val, kd, A, p, s_offset.val)
                x_line, y_line = target_line(A, p, s_offset.val, x_max=15.0, straight_length=p)
                y_interp = np.interp(xs_h, x_line, y_line)
                std_dev = np.std(ys_h - y_interp)
                final_x = xs_h[-1] if len(xs_h) > 0 else 0.01
                scores.append(std_dev / final_x)

        avg = np.mean(scores)
        avg_score_kd.append(avg)

        # --- UPDATE PLOT LIVE ---
        heat_live[:, j] = avg  # fill column
        heatmap_img.set_data(heat_live)
        heatmap_img.set_clim(np.min(heat_live), np.max(heat_live) + 1e-6)
        fig.canvas.draw_idle()
        plt.pause(0.001)

    best_kd = kd_values[np.argmin(avg_score_kd)]

    # Update sliders
    s_kp.set_val(best_kp)
    s_ki.set_val(Ki_val)
    s_kd.set_val(best_kd)

    print(f"Auto-calibration finished:\nBest Kp = {best_kp:.3f}, Ki = {Ki_val:.3f}, Kd = {best_kd:.3f}")
# --- Connect ---
s_kp.on_changed(update_path)
s_ki.on_changed(update_path)
s_kd.on_changed(update_path)
s_amp.on_changed(update_path)
s_period.on_changed(update_path)
s_offset.on_changed(update_path)
b_update.on_clicked(update_heatmap)
b_auto.on_clicked(auto_calibrate)

# --- Initial plot ---
update_path(None)
plt.show()