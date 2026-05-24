import cv2
import numpy as np
import robotica

"""
FMS + WALL SAFETY + BOLA RÁPIDA ADAPTATIVA + LOST MODE ACTIVO
"""

# -----------------------
# DETECCIÓN DE BOLA
# -----------------------
def detect_ball(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (0,120,70), (10,255,255)) | \
           cv2.inRange(hsv, (170,120,70), (180,255,255))

    area = cv2.countNonZero(mask)

    if area < 25:
        return None

    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    return cx, img.shape[1], area


# -----------------------
# MEMORIA GLOBAL
# -----------------------
smooth_cx = None
alpha = 0.6

state = "FOLLOW"
last_area = None
last_cx_fast = None

# 🔥 NUEVO LOST MODE
lost_timer = 0
lost_mode = False
last_turn_dir = 1


# -----------------------
# FMS
# -----------------------
def get_state(area, sonar):
    global state, last_area

    front = min(sonar[3:6])
    left = min(sonar[6:10])
    right = min(sonar[0:3])

    wall_risk = (front < 0.25) or (left < 0.22) or (right < 0.22)

    if last_area is None:
        last_area = area
        return "FOLLOW"

    change = (area - last_area) / (last_area + 1)
    last_area = area

    if front < 0.22:
        state = "EMERGENCY"
        return state

    if state == "WAIT":
        if area < 42000:
            state = "FOLLOW"
        return state

    if area > 50000 and abs(change) < 0.01:
        state = "WAIT"
        return state

    if wall_risk:
        state = "CAREFUL"
        return state

    state = "FOLLOW"
    return state


# -----------------------
# WALL SAFETY SUAVE
# -----------------------
def wall_soft_bias(sonar):
    front = min(sonar[3:6])
    left  = min(sonar[6:10])
    right = min(sonar[0:3])

    if front < 0.22:
        return -2.2, 2.2

    front_risk = np.clip(0.5 - front, 0, 0.5)
    left_risk  = np.clip(0.6 - left, 0, 0.6)
    right_risk = np.clip(0.6 - right, 0, 0.6)

    total_risk = front_risk + left_risk + right_risk

    base = 1.7 - 2.2 * total_risk
    base = np.clip(base, 0.25, 1.7)

    bias = (right_risk - left_risk)

    turn = 2.0 * bias
    turn = np.clip(turn, -0.8, 0.8)

    if left < 0.35:
        turn += 0.4
    if right < 0.35:
        turn -= 0.4

    return base - turn, base + turn


# -----------------------
# FOLLOW BOLA
# -----------------------
def follow_ball(ball, width, sonar, ball_fast):
    global smooth_cx, state, last_turn_dir

    center = width // 2

    if ball:
        cx, w, area = ball

        if smooth_cx is None:
            smooth_cx = cx
        else:
            smooth_cx = int(alpha * cx + (1 - alpha) * smooth_cx)

        error = (center - smooth_cx) / center
        if abs(error) < 0.08:
            error = 0

        last_turn_dir = np.sign(error)

        state = get_state(area, sonar)

        if state == "EMERGENCY":
            return -1.0, -1.0

        if state == "WAIT":
            return 0.0, 0.0

        target_area = 55000

        dist_error = np.log((target_area + 1) / (area + 1))
        dist_error = np.clip(dist_error, -2.0, 2.0)

        base = 1.8 + 1.6 * np.tanh(dist_error)
        base = np.clip(base, 0.2, 4.0)

        turn = 1.1 * error
        turn = np.clip(turn, -1.2, 1.2)

        base *= np.clip(1 - 0.2 * abs(error), 0.6, 1.0)

        if state == "CAREFUL":
            base *= 0.7
            turn *= 0.6

        return base - turn, base + turn

    return 0.0, 0.0


# -----------------------
# LOST MODE (🔥 NUEVO)
# -----------------------
def lost_behavior():
    global lost_timer, last_turn_dir

    lost_timer += 1

    # exploración activa
    spin = 1.2 * last_turn_dir

    # cada mucho tiempo cambia dirección
    if lost_timer > 25:
        last_turn_dir *= -1
        lost_timer = 0

    return 0.6 - spin, 0.6 + spin


# -----------------------
# MAIN
# -----------------------
def main():
    global last_cx_fast, lost_mode, lost_timer

    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', use_camera=True)

    coppelia.start_simulation()

    try:
        while coppelia.is_running():

            img = robot.get_image()
            sonar = robot.get_sonar()
            ball = detect_ball(img)

            # -----------------------
            # DETECCIÓN BOLA RÁPIDA
            # -----------------------
            ball_fast = False

            if ball:
                cx, w, area = ball

                if last_cx_fast is not None:
                    if abs(cx - last_cx_fast) > 18:
                        ball_fast = True

                last_cx_fast = cx
                lost_mode = False
                lost_timer = 0

            else:
                lost_mode = True

            # -----------------------
            # CONTROL
            # -----------------------
            if lost_mode:
                ball_cmd = lost_behavior()
            else:
                ball_cmd = follow_ball(ball, img.shape[1], sonar, ball_fast)

            wall_cmd = wall_soft_bias(sonar)

            # -----------------------
            # MEZCLA
            # -----------------------
            if ball_fast:
                w_ball = 0.6
                w_wall = 0.4
                ball_cmd = (ball_cmd[0] * 0.8, ball_cmd[1] * 0.8)
            else:
                w_ball = 0.75
                w_wall = 0.25

            left = w_ball * ball_cmd[0] + w_wall * wall_cmd[0]
            right = w_ball * ball_cmd[1] + w_wall * wall_cmd[1]

            robot.set_speed(left, right)

            cv2.imshow("camera", img)
            cv2.waitKey(1)

    finally:
        coppelia.stop_simulation()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()