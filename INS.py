import tkinter
from tkinter import ttk
import numpy as np
import datetime
import pygame
import time


##################################################################
# FUNCTIONS
def draw_text(text, x, y):
    font = pygame.font.SysFont("Ubuntu Mono", 30)
    text_colour = (255, 255, 255)
    image = font.render(text, True, text_colour)
    screen.blit(image, (x, y))


def draw_ins_text(text, x, y):
    font = pygame.font.SysFont("Ubuntu Mono", 35)
    text_colour = (255, 255, 255)
    image = font.render(text, True, text_colour)
    screen.blit(image, (x, y))


class Button:
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.clicked = False

    def draw(self):
        action = False
        mouse_position = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_position):
            if pygame.mouse.get_pressed()[0] == 1\
                    and self.clicked is False:
                self.clicked = True
                action = True

        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False

        screen.blit(self.image, (self.rect.x, self.rect.y))
        return action


def time_zone(zone, time_type):
    timezone = datetime.timedelta(hours=zone)
    daylight = datetime.timedelta(hours=1)
    utc_time = datetime.datetime.now(datetime.timezone.utc)

    if time_type == "UTC":
        display_time = utc_time.strftime("%d %B %H:%M:%S UTC")
    elif time_type == "LOC":
        local_time = utc_time + timezone
        display_time = local_time.strftime("%d %B %H:%M:%S LOC")
    else:
        local_dst_time = utc_time + timezone + daylight
        display_time =\
            local_dst_time.strftime("%d %B %H:%M:%S LST")

    return display_time


def enter_initial_data():
    init_data.clear()
    y_sign = y_sign_combobox.get()
    x_sign = x_sign_combobox.get()

    y_pos_10m = int(y_pos_10m_spinbox.get())
    y_pos_1m = int(y_pos_1m_spinbox.get())
    y_pos_10cm = int(y_pos_10cm_spinbox.get())
    y_pos_1cm = int(y_pos_1cm_spinbox.get())

    x_pos_10m = int(x_pos_10m_spinbox.get())
    x_pos_1m = int(x_pos_1m_spinbox.get())
    x_pos_10cm = int(x_pos_10cm_spinbox.get())
    x_pos_1cm = int(x_pos_1cm_spinbox.get())

    y_pos = \
        y_pos_10m*10 + y_pos_1m + y_pos_10cm*0.1 + y_pos_1cm*0.01
    x_pos = \
        x_pos_10m*10 + x_pos_1m + x_pos_10cm*0.1 + x_pos_1cm*0.01

    match y_sign:
        case "-":
            y_pos *= -1

    match x_sign:
        case "-":
            x_pos *= -1

    y_pos = y_pos
    x_pos = x_pos

    time_zone_p = 0
    match time_zone_combobox.get():
        case "+4:30":
            time_zone_p = str(4.5)
        case "+4:00":
            time_zone_p = str(4)
        case "+3:30":
            time_zone_p = str(3.5)
        case "+3:00":
            time_zone_p = str(3)
        case "+2:00":
            time_zone_p = str(2)
        case "+1:00":
            time_zone_p = str(1)
        case "+0:00":
            time_zone_p = str(0)
        case "-1:00":
            time_zone_p = str(-1)
        case "-2:00":
            time_zone_p = str(-2)
        case "-3:00":
            time_zone_p = str(-3)
        case "-3:30":
            time_zone_p = str(-3.5)
        case "-4:00":
            time_zone_p = str(-4)

    time_UTC = datetime.datetime.now(datetime.timezone.utc)
    time_start = time_UTC.strftime("%d %B %Y %H:%M:%S UTC")
    initial_data_list = ["Y_0: "+str("%+06.2f" % y_pos+" m"),
                         "X_0: "+str("%+06.2f" % x_pos+" m"),
                         "TZ: "+str(time_zone_combobox.get()),
                         "T_0: "+str(time_start)]
    initial_data = open("VDR/initial_data.txt", "w")
    initial_data_content = "\n".join(initial_data_list)
    initial_data.writelines(initial_data_content)
    init_data.append(y_pos)
    init_data.append(x_pos)
    init_data.append(float(time_zone_p))
    insert_mode.destroy()


def mag_calibrated(m_vector):
    A_inv = np.array([[0.02510398, 0.00009885, 0.00010111],
                      [0.00009885, 0.02606499, 0.00006862],
                      [0.00010111, 0.00006862, 0.02398707]])

    b = np.array([[-16.71351618], [-23.28736059], [8.9439934]])
    h_m = A_inv @ (m_vector - b)

    A_inv_2 = np.array([[0.97846124, -0.00058379, 0.00173339],
                        [-0.00058379, 0.98458849, 0.00436829],
                        [0.00173339, 0.00436829, 0.98310266]])

    b_2 = np.array([[0.00852009], [-0.23644724], [-0.02024542]])

    h_m2 = A_inv_2 @ (h_m - b_2)

    mag_x = -h_m2[1][0]
    mag_y = -h_m2[0][0]
    mag_z = h_m2[2][0]
    h = np.array([[mag_x], [mag_y], [mag_z]])

    return h


def acc_calibrated(a_vector):
    A_inv = np.array([[0.996258, -0.000023, -0.004603],
                      [-0.000023, 0.998775, -0.001194],
                      [-0.004603, -0.001194, 0.986073]])

    b = np.array([[0.02408251], [0.0190393], [-0.1695812]])

    a_vector_cal_1 = 9.81464715 * (A_inv @ (a_vector - b))

    sin_phi = 0.06801527
    cos_phi = 0.99768428
    sin_theta = -0.07281379
    cos_theta = 0.99734555
    sin_psi = -0.02265539
    cos_psi = 0.99974333

    rot_phi = np.array([[1, 0, 0],
                        [0, cos_phi, -sin_phi],
                        [0, sin_phi, cos_phi]])
    rot_theta = np.array([[cos_theta, 0, sin_theta],
                          [0, 1, 0],
                          [-sin_theta, 0, cos_theta]])
    rot_psi = np.array([[cos_psi, -sin_psi, 0],
                        [sin_psi, cos_psi, 0],
                        [0, 0, 1]])

    a_vector_cal = rot_psi @ rot_theta @ rot_phi @ a_vector_cal_1
    return a_vector_cal


def gyro_calibrated(g_vector):
    b = np.array([[-0.00421685], [0.00510555], [0.00040773]])

    g_vector_cal = 0.0174532925 * g_vector + b
    return g_vector_cal


def att_trig(attitude_v):
    phi = attitude_v[0][0]
    theta = attitude_v[1][0]
    psi = attitude_v[2][0]

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    trig = sin_phi, cos_phi, sin_theta, cos_theta, tan_theta, \
        sin_psi, cos_psi
    return trig


def acceleration_true(attitude_trig, a_vector):
    sin_phi = attitude_trig[0]
    cos_phi = attitude_trig[1]
    sin_theta = attitude_trig[2]
    cos_theta = attitude_trig[3]
    g_b = 9.81464715 * np.array([[sin_theta],
                                 [-sin_phi * cos_theta],
                                 [-cos_phi * cos_theta]])
    acc_true_n = -1*(a_vector - g_b)
    return acc_true_n


def mag_rotate(attitude_trig, mag_rdg):
    sin_phi = attitude_trig[0]
    cos_phi = attitude_trig[1]
    sin_theta = attitude_trig[2]
    cos_theta = attitude_trig[3]

    rot_phi = np.array([[1, 0, 0],
                        [0, cos_phi, sin_phi],
                        [0, -sin_phi, cos_phi]])
    rot_theta = np.array([[cos_theta, 0, -sin_theta],
                          [0, 1, 0],
                          [sin_theta, 0, cos_theta]])

    rot = rot_phi @ rot_theta
    mag_2 = rot @ mag_rdg
    return mag_2


def acc_attitude(acc):
    ax_1 = acc[0][0]
    ay_1 = acc[1][0]
    az_1 = acc[2][0]

    if ax_1 > 9.81464715:
        pitch_acc = np.pi / 2
    elif ax_1 < -9.81464715:
        pitch_acc = -np.pi / 2
    else:
        pitch_acc = np.arcsin(ax_1 / 9.81464715)
    roll_acc = np.arctan2(ay_1, az_1) + np.pi
    attitude_acc = np.array([[roll_acc], [pitch_acc]])

    if attitude_acc[0][0] > np.pi:
        attitude_acc = attitude_acc - np.array([[2 * np.pi], [0]])
    else:
        attitude_acc = attitude_acc

    return attitude_acc


def body2nav(attitude_trig, a_vector):
    sin_phi = attitude_trig[0]
    cos_phi = attitude_trig[1]
    sin_theta = attitude_trig[2]
    cos_theta = attitude_trig[3]
    sin_psi = attitude_trig[5]
    cos_psi = attitude_trig[6]

    rot_phi = np.array([[1, 0, 0],
                        [0, cos_phi, -sin_phi],
                        [0, sin_phi, cos_phi]])
    rot_theta = np.array([[cos_theta, 0, sin_theta],
                          [0, 1, 0],
                          [-sin_theta, 0, cos_theta]])
    rot_psi = np.array([[cos_psi, -sin_psi, 0],
                        [sin_psi, cos_psi, 0],
                        [0, 0, 1]])

    vector_rotated = rot_psi @ rot_theta @ rot_phi @ a_vector
    return vector_rotated


def ekf(attitude, gyro_reading, acc_reading, mag_reading,
        covariance, acc_att_then, delta_time):
    def gyro_rate2att_rate(attitude_trig, gyro_rates):
        sin_phi = attitude_trig[0]
        cos_phi = attitude_trig[1]
        cos_theta = attitude_trig[3]
        tan_theta = attitude_trig[4]

        rot = np.array([[1, sin_phi * tan_theta,
                         cos_phi * tan_theta],
                        [0, cos_phi, -sin_phi],
                        [0, sin_phi / cos_theta,
                         cos_phi / cos_theta]])

        att_rate = rot @ gyro_rates
        return att_rate

    def jacob_f(attitude_trig, gyro_rates):
        q = gyro_rates[1][0]
        r = gyro_rates[2][0]

        sin_phi = attitude_trig[0]
        cos_phi = attitude_trig[1]
        sin_theta = attitude_trig[2]
        cos_theta = attitude_trig[3]
        tan_theta = attitude_trig[4]

        j_11 = (q * cos_phi - r * sin_phi) * tan_theta
        j_12 = (q * sin_phi + r * cos_phi) / (cos_theta ** 2)
        j_13 = 0

        j_21 = -q * sin_phi - r * cos_phi
        j_22 = 0
        j_23 = 0

        j_31 = (q * cos_phi - r * sin_phi) / cos_theta
        j_32 = (q * sin_phi + r * cos_phi) * sin_theta / \
               (cos_theta ** 2)
        j_33 = 0

        jacob = np.array([[j_11, j_12, j_13],
                          [j_21, j_22, j_23],
                          [j_31, j_32, j_33]])
        return jacob

    def jacob_h1(attitude_trig):
        sin_phi = attitude_trig[0]
        cos_phi = attitude_trig[1]
        sin_theta = attitude_trig[2]
        cos_theta = attitude_trig[3]

        j_11 = 0
        j_12 = cos_theta
        j_13 = 0

        j_21 = -cos_phi * cos_theta
        j_22 = sin_phi * sin_theta
        j_23 = 0

        j_31 = sin_phi * cos_theta
        j_32 = cos_phi * sin_theta
        j_33 = 0

        jacob = 9.81464715 * np.array([[j_11, j_12, j_13],
                                       [j_21, j_22, j_23],
                                       [j_31, j_32, j_33]])
        return jacob

    def acc_model(attitude_trig):
        sin_phi = attitude_trig[0]
        cos_phi = attitude_trig[1]
        sin_theta = attitude_trig[2]
        cos_theta = attitude_trig[3]

        body_g = np.array([[sin_theta],
                           [-sin_phi * cos_theta],
                           [-cos_phi * cos_theta]])

        grav_vector = 9.81464715 * body_g
        return grav_vector

    acc_att_now = 0.2*acc_attitude(acc_reading) + 0.8*acc_att_then

    Q_1 = acc_att_now - np.array([[attitude[0][0]],
                                  [attitude[1][0]]])
    Q_phi = (np.degrees(Q_1[0][0]))**2 / 900
    Q_theta = (np.degrees(Q_1[1][0]))**2 / 900

    Q_2 = np.array([[Q_phi, 0, 0],
                    [0, Q_theta, 0],
                    [0, 0, 0]])

    Q_coefficient = (0.0698 * delta_time)**2  # (3deg_err * dt)^2
    Q = Q_coefficient * np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]]) + Q_2

    R1_coefficient = 0.02**2 * 97441  # 3deg_err * coefficient
    R1 = R1_coefficient * np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])

    att_trig_predict = att_trig(attitude)

    # PREDICT STEP
    attitude_prior = attitude + delta_time * \
        gyro_rate2att_rate(att_trig_predict, gyro_reading)
    F = jacob_f(att_trig_predict, gyro_reading)
    P_prior = covariance + delta_time * \
        (F @ covariance + covariance @ np.transpose(F) + Q)

    # WAIT TIME

    att_trig_update = att_trig(attitude_prior)

    # UPDATE STEP
    z1 = acc_reading - acc_model(att_trig_update)
    H1 = jacob_h1(att_trig_update)
    S1 = H1 @ P_prior @ np.transpose(H1) + R1

    K1 = P_prior @ np.transpose(H1) @ np.linalg.inv(S1)

    attitude_posterior = attitude_prior + (
            K1 @ z1)  # NO HEADING

    I_matrix = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])

    P_posterior = (I_matrix - K1 @ H1) @ P_prior

    # complementary filter

    phi_2 = attitude_posterior[0][0]
    theta_2 = attitude_posterior[1][0]
    heading_gyro = attitude_prior[2][0]

    att_trig_mag = att_trig(attitude_posterior)
    mag_rotated = mag_rotate(att_trig_mag, mag_reading)
    m_y = mag_rotated[1][0]
    m_x = mag_rotated[0][0]

    heading_mag = np.arctan2(-m_y, m_x)

    if np.abs(heading_mag - heading_gyro) <= np.radians(10):
        weight = 0.01
    elif np.abs(heading_mag - heading_gyro) > np.radians(90):
        weight = 0.81
    else:
        weight = np.degrees(np.abs(heading_mag -
                                   heading_gyro))**2/10000

    psi_2 = (1-weight) * heading_gyro + weight * heading_mag

    attitude_2 = np.array([[phi_2], [theta_2], [psi_2]])

    return attitude_2, P_posterior, acc_att_now


def velocity_vector(vector, limit, up, hdg):
    vector_knot = 1.9438445 * np.array([[1, 0], [0, -1]]) @ vector
    vector_drawn = 0
    vec_3 = 0

    def vector_rot(vec1, hdg1):
        sin_hdg = np.sin(hdg1)
        cos_hdg = np.cos(hdg1)

        rot = np.array([[cos_hdg, sin_hdg],
                        [-sin_hdg, cos_hdg]])
        v_vector_rot = rot @ vec1
        return v_vector_rot

    match limit:
        case 1:
            vector_drawn = vector_knot * 140
        case 2:
            vector_drawn = vector_knot / 2 * 140
        case 3:
            vector_drawn = vector_knot / 4 * 140
        case 4:
            vector_drawn = vector_knot / 8 * 140
        case 5:
            vector_drawn = vector_knot / 16 * 140

    v_len = np.sqrt(vector_drawn[0][0]**2 + vector_drawn[1][0]**2)

    if v_len > 140:
        colour = (225, 255, 255)
        s = 140 / v_len
        vec_2 = vector_drawn * s
    else:
        colour = (225, 5, 0)
        vec_2 = vector_drawn

    match up:
        case "head":
            vec_3 = vector_rot(vec_2, hdg)
        case "north":
            vec_3 = vec_2

    x = vec_3[0][0] + 570
    y = vec_3[1][0] + 220
    return x, y, colour


def rotate_heading_vector(hdg):
    vector = np.array([[0], [-140]])
    sin_hdg = np.sin(hdg)
    cos_hdg = np.cos(hdg)

    rot = np.array([[cos_hdg, -sin_hdg],
                    [sin_hdg, cos_hdg]])
    v_vector_rot = rot @ vector
    return v_vector_rot


init_data = []
init_att_hdg = []
init_acc = []
init_mag = []
##################################################################
# GUIS

# TKINTER GUI
insert_mode = tkinter.Tk()
insert_mode.geometry("700x360")
insert_mode.title("INSERT MODE")

# INPUT
frame = tkinter.Frame(insert_mode)
frame.pack()

position_frame = tkinter.LabelFrame(frame,
                                    text="Initial Position")
position_frame.grid(row=0, column=0, padx=5, pady=5)
frame.columnconfigure(0, minsize=10)


pos_1m_label = tkinter.Label(position_frame, text="metres")
pos_1m_label.grid(row=1, column=3)
pos_1cm_label = tkinter.Label(position_frame, text="centimetres")
pos_1cm_label.grid(row=1, column=5)
sign_label = tkinter.Label(position_frame, text="Sign")
sign_label.grid(row=1, column=1)


# Y POSITION
y_sign_combobox = ttk.Combobox(position_frame, values=["+", "-"],
                               width=10)
y_sign_combobox.grid(row=2, column=1)

y_pos_label = tkinter.Label(position_frame, text="Y")
y_pos_label.grid(row=2, column=0)

y_pos_10m_spinbox = tkinter.Spinbox(position_frame, from_=0, to=9,
                                    width=10)
y_pos_10m_spinbox.grid(row=2, column=2)

y_pos_1m_spinbox = tkinter.Spinbox(position_frame, from_=0, to=9,
                                   width=10)
y_pos_1m_spinbox.grid(row=2, column=3)

y_pos_10cm_spinbox = tkinter.Spinbox(position_frame,
                                     from_=0, to=9, width=10)
y_pos_10cm_spinbox.grid(row=2, column=4)

y_pos_1cm_spinbox = tkinter.Spinbox(position_frame, from_=0, to=9,
                                    width=10)
y_pos_1cm_spinbox.grid(row=2, column=5)

# X POSITION
x_sign_combobox = ttk.Combobox(position_frame, values=["+", "-"],
                               width=10)
x_sign_combobox.grid(row=3, column=1)

x_pos_label = tkinter.Label(position_frame, text="X")
x_pos_label.grid(row=3, column=0)

x_pos_10m_spinbox = tkinter.Spinbox(position_frame, from_=0, to=9,
                                    width=10)
x_pos_10m_spinbox.grid(row=3, column=2)

x_pos_1m_spinbox = tkinter.Spinbox(position_frame, from_=0, to=9,
                                   width=10)
x_pos_1m_spinbox.grid(row=3, column=3)

x_pos_10cm_spinbox = tkinter.Spinbox(position_frame,
                                     from_=0, to=9, width=10)
x_pos_10cm_spinbox.grid(row=3, column=4)

x_pos_1cm_spinbox = tkinter.Spinbox(position_frame, from_=0, to=9,
                                    width=10)
x_pos_1cm_spinbox.grid(row=3, column=5)

for widget in position_frame.winfo_children():
    widget.grid_configure(padx=5, pady=5)


time_zone_frame = tkinter.LabelFrame(frame, text="Time Zone")
time_zone_frame.grid(row=1, column=0, sticky="news", padx=5,
                     pady=5)

time_zone_combobox = \
    ttk.Combobox(time_zone_frame,
                 values=["+4:30", "+4:00", "+3:30", "+3:00",
                         "+2:00", "+1:00", "+0:00", "-1:00",
                         "-2:00", "-3:00", "-3:30", "-4:00"],
                 width=10)
time_zone_combobox.grid(row=0, column=0)

for widget in time_zone_frame.winfo_children():
    widget.grid_configure(padx=5, pady=5)


entry_button = tkinter.Button(frame, text="START NAVIGATION",
                              command=enter_initial_data)
entry_button.grid(row=3, column=0)

insert_mode.mainloop()

# PYGAME GUI
pygame.init()
screen = pygame.display.set_mode((740, 400))
icon = pygame.image.load("images/icon_png.png")
pygame.display.set_icon(icon)

time_button_image = pygame.image.load("images/time_button.png").\
    convert_alpha()
time_button = Button(280, 15, time_button_image)

exit_button_1_image = pygame.image.load("images/exit_1.png").\
    convert_alpha()
exit_button_1 = Button(15, 350, exit_button_1_image)

exit_screen = pygame.image.load("images/exit_screen.png").\
    convert_alpha()
exit_screen = Button(234, 147, exit_screen)

exit_button_yes_image = pygame.image.load("images/yes_rect.png").\
    convert_alpha()
exit_button_yes = Button(267, 198, exit_button_yes_image)

exit_button_no_image = pygame.image.load("images/no_rect.png").\
    convert_alpha()
exit_button_no = Button(415, 198, exit_button_no_image)

initialization_image = pygame.image.\
    load("images/initialization.png").convert_alpha()
initialization_button = Button(48, 172, initialization_image)

head_up_image = pygame.image.load("images/H.png").convert_alpha()
head_up_button = Button(668, 80, head_up_image)

north_up_image = pygame.image.load("images/N.png").convert_alpha()
north_up_button = Button(668, 80, north_up_image)

knot_1_image = pygame.image.load("images/1.png").convert_alpha()
knot_1_button = Button(698, 80, knot_1_image)

knot_2_image = pygame.image.load("images/2.png").convert_alpha()
knot_2_button = Button(698, 80, knot_2_image)

knot_4_image = pygame.image.load("images/4.png").convert_alpha()
knot_4_button = Button(698, 80, knot_4_image)

knot_8_image = pygame.image.load("images/8.png").convert_alpha()
knot_8_button = Button(698, 80, knot_8_image)

knot_16_image = pygame.image.load("images/16.png").convert_alpha()
knot_16_button = Button(698, 80, knot_16_image)

knot_up_image = pygame.image.load("images/knot_up.png").\
    convert_alpha()
knot_up_button = Button(400, 307, knot_up_image)

knot_down_image = pygame.image.load("images/knot_down.png").\
    convert_alpha()
knot_down_button = Button(400, 349, knot_down_image)

vector_image = pygame.image.load("images/vector.png").\
    convert_alpha()
vector_button = Button(472, 121, vector_image)

##################################################################
# PROGRAM
time_connect = time.time()
start_bool = False
while (time.time() - time_connect) < 5:
    try:
        from mpu9250_i2c import *
        start_bool = True
        break
    except ValueError:
        continue


def navigation_mode_gui():  # NAVIGATION MODE SCREEN
    pygame.display.set_caption("NAVIGATION MODE")

    dt = 0.035  # s

    P_coefficient = 0.08726**2
    P = P_coefficient * np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])

    gyro_prev_fil = np.array([[0], [0], [0]])
    acc_prev_fil = init_acc[0]
    mag_prev_fil = init_mag[0]

    acc_prev = np.array([[0], [0], [0]])
    velocity_prev = np.array([[0], [0], [0]])

    # VESSEL DATA READ
    att_hdg = init_att_hdg[0]
    att_acc_prev = np.array([[att_hdg[0][0]], [att_hdg[1][0]]])
    pos_prev = np.array([[init_data[1]], [init_data[0]], [0]])

    time_type = str("UTC")
    up_type = str("head")
    exit_stage = str("0")

    knot_limit = 1

    run = True
    while run:
        time_stamp = time.time()
        screen.fill((69, 69, 69))

        # MEASUREMENTS
        ax, ay, az, wx, wy, wz = mpu6050_conv()
        mx, my, mz = AK8963_conv()

        # AHRS
        acc_measured = np.array([[ax], [ay], [-az]])
        acc_cal = acc_calibrated(acc_measured)
        mag_measured = np.array([[mx], [my], [mz]])
        mag_cal = mag_calibrated(mag_measured)
        gyro_measured = np.array([[-wx], [-wy], [wz]])
        gyro_cal = gyro_calibrated(gyro_measured)

        gyro_fil = 0.99 * gyro_cal + 0.01 * gyro_prev_fil
        acc_fil = 0.9 * acc_cal + 0.1 * acc_prev_fil
        mag_fil = 0.8 * mag_cal + 0.2 * mag_prev_fil

        EKF = ekf(att_hdg, gyro_prev_fil, acc_fil, mag_fil, P,
                  att_acc_prev, dt)

        att_hdg_1 = EKF[0]
        P = EKF[1]
        att_acc_prev = EKF[2]

        gyro_prev_fil = gyro_fil
        acc_prev_fil = acc_fil
        mag_prev_fil = mag_fil

        if att_hdg_1[0][0] < -np.pi:
            att_hdg_1 = att_hdg_1 + \
                        np.array([[2 * np.pi], [0], [0]])
        elif att_hdg_1[0][0] > np.pi:
            att_hdg_1 = att_hdg_1 - \
                        np.array([[2 * np.pi], [0], [0]])
        elif att_hdg_1[1][0] < -np.pi / 2:
            att_hdg_1 = -np.array([[0], [np.pi], [0]]) - att_hdg_1
        elif att_hdg_1[1][0] > np.pi / 2:
            att_hdg_1 = np.array([[0], [np.pi], [0]]) - att_hdg_1
        elif att_hdg_1[2][0] > np.pi:
            att_hdg_1 = att_hdg_1-np.array([[0], [0], [2*np.pi]])
        elif att_hdg_1[2][0] < -np.pi:
            att_hdg_1 = att_hdg_1+np.array([[0], [0], [2*np.pi]])
        else:
            att_hdg_1 = att_hdg_1

        att_hdg = att_hdg_1
        att_hdg_n = att_hdg_1 - \
            np.array([[0], [0], [0.11490084]]) - \
            np.array([[0], [0], [0.43633231]])  # var, err

        att_hdg_NED = np.array([[1, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1]]) @ att_hdg_n

        if att_hdg_NED[2][0] < 0:
            att_hdg_NED = att_hdg_NED + \
                          np.array([[0], [0], [2*np.pi]])
        else:
            att_hdg_NED = att_hdg_NED

        roll = "ROL: " + \
            '{:+006.1f}'.format(np.degrees(att_hdg_NED[0][0]))+"°"
        pitch = "PCH:  " + \
            '{:+05.1f}'.format(np.degrees(att_hdg_NED[1][0]))+"°"
        heading = att_hdg_NED[2][0]
        heading_deg = "HDG:  " + \
            '{:005.1f}'.format(np.degrees(heading))+"°"

        # INS
        body2nav_trig = att_trig(att_hdg_n)
        acc_b_true = acceleration_true(body2nav_trig, acc_cal)
        acc_n_true = body2nav(body2nav_trig, acc_b_true)

        acc_now = acc_n_true
        delta_v = (acc_now + acc_prev) * dt/2
        velocity_now = velocity_prev + delta_v

        delta_pos_1 = (velocity_now + velocity_prev) * dt/2
        delta_pos = np.array([[-delta_pos_1[1][0]],
                              [delta_pos_1[0][0]],
                              [delta_pos_1[2][0]]])

        pos_now = pos_prev + delta_pos
        pos_x = "X: "+'{:+06.2f}'.format(pos_now[0][0])+" m"
        pos_y = "Y: "+'{:+06.2f}'.format(pos_now[1][0])+" m"

        acc_prev = acc_now
        velocity_prev = velocity_now
        pos_prev = pos_now

        v_vector = np.array([[-velocity_now[1][0]],
                             [velocity_now[0][0]]])

        COG_1 = np.degrees(np.arctan2(v_vector[0][0],
                                      v_vector[1][0]))
        if COG_1 < 0:
            COG_1 += 360

        COG = "COG:  "+'{:005.1f}'.format(COG_1)+"°"
        vel = "V: "+'{:04.1f}'.format(1.9438445*np.sqrt(
            velocity_now[0][0]**2 + velocity_now[1][0]**2))+" kn"

        # GUI RENDER

        # crosshair
        pygame.draw.line(screen, (255, 255, 255), (570, 80),
                         (570, 360), width=2)
        pygame.draw.line(screen, (255, 255, 255), (430, 220),
                         (710, 220), width=2)

        match up_type:  # heading line
            case "north":
                hdg = rotate_heading_vector(heading)
                pygame.draw.line(screen, (0, 128, 0),
                                 (570, 220),
                                 (570+hdg[0][0], 220+hdg[1][0]),
                                 width=2)

        # circles
        pygame.draw.circle(screen, (255, 255, 255), (570, 220),
                           radius=140, width=2)
        pygame.draw.circle(screen, (255, 255, 255), (570, 220),
                           radius=70, width=2)

        # velocity vector
        v_vector_drawn = velocity_vector(v_vector, knot_limit,
                                         up_type, heading)
        pygame.draw.line(screen, v_vector_drawn[2], (570, 220),
                         (v_vector_drawn[0], v_vector_drawn[1]),
                         width=5)

        draw_text(time_zone(init_data[2], time_type), 280, 30)
        draw_ins_text(roll, 15, 30)
        draw_ins_text(pitch, 15, 70)
        draw_ins_text(heading_deg, 15, 110)
        draw_ins_text(COG, 15, 150)
        draw_ins_text(pos_y, 15, 210)
        draw_ins_text(pos_x, 15, 250)
        draw_ins_text(vel, 15, 290)

        if time_button.draw():
            match time_type:
                case "UTC":
                    time_type = str("LOC")
                case "LOC":
                    time_type = str("LST")
                case "LST":
                    time_type = str("UTC")

        match up_type:
            case "head":
                head_up_button.draw()
            case "north":
                north_up_button.draw()

        match knot_limit:
            case 1:
                knot_1_button.draw()
            case 2:
                knot_2_button.draw()
            case 3:
                knot_4_button.draw()
            case 4:
                knot_8_button.draw()
            case 5:
                knot_16_button.draw()

        if vector_button.draw():
            match up_type:
                case "head":
                    up_type = str("north")
                case "north":
                    up_type = str("head")

        if knot_up_button.draw():
            if knot_limit == 5:
                knot_limit = 1
            else:
                knot_limit += 1

        if knot_down_button.draw():
            if knot_limit == 1:
                knot_limit = 5
            else:
                knot_limit -= 1

        if exit_button_1.draw():
            exit_stage = str("1")

        match exit_stage:
            case "1":
                exit_screen.draw()

                if exit_button_yes.draw():
                    run = False
                if exit_button_no.draw():
                    exit_stage = "0"

                draw_text("Are you sure?", 264, 153)
                draw_text("YES", 277, 200)
                draw_text("NO", 424, 200)

        draw_text("EXIT", 25, 350)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        loop_time = time_stamp - time.time()
        pygame.display.update()
        time.sleep(dt - loop_time)


def initialization_mode_gui():  # INITIALIZATION MODE SCREEN
    pygame.display.set_caption("NAVIGATION MODE")
    screen.fill((69, 69, 69))
    initialization_button.draw()

    acc_sum = np.array([[0], [0], [0]])
    mag_sum = np.array([[0], [0], [0]])

    i = 1
    initialization_stamp = time.time()
    run = True
    while run:
        ax, ay, az, wx, wy, wz = mpu6050_conv()
        mx, my, mz = AK8963_conv()

        acc_measured = np.array([[ax], [ay], [-az]])
        acc_cal = acc_calibrated(acc_measured)
        mag_measured = np.array([[mx], [my], [mz]])
        mag_cal = mag_calibrated(mag_measured)

        acc_sum = acc_sum + acc_cal
        mag_sum = mag_sum + mag_cal

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        if time.time() - initialization_stamp > 5:
            acc_mean = acc_sum / i
            mag_mean = mag_sum / i

            init_att_1 = acc_attitude(acc_mean)
            init_att_2 = np.array([[init_att_1[0][0]],
                                   [init_att_1[1][0]], [0]])

            init_trig = att_trig(init_att_2)
            init_mag_rotate = mag_rotate(init_trig, mag_mean)
            m_y = init_mag_rotate[1][0]
            m_x = init_mag_rotate[0][0]

            init_heading_mag = np.arctan2(-m_y, m_x)
            init_hdg_vector = np.array([[0], [0],
                                        [init_heading_mag]])
            init_att_hdg_v = init_att_2 + init_hdg_vector
            init_att_hdg.append(init_att_hdg_v)
            init_acc.append(acc_mean)
            init_mag.append(mag_mean)

            run = False
            navigation_mode_gui()

        pygame.display.update()
        i += 1
        time.sleep(0.01)


initialization_mode_gui()
