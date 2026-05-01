import cv2
import numpy as np
import robotica

"""
ARQUITECTURA DE BROOKS REAL (MEJORADA)
"""

# =========================
# DETECCIÓN DE BOLA
# =========================
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


# =========================
# MEMORIA
# =========================
last_cx = None
last_seen_time = 0
smooth_cx = None
alpha = 0.6

corner_mode = False
corner_dir = 1


# =========================
# ESQUINA
# =========================
def is_corner(front, left, right):
    return front < 0.45 and (left < 0.5 or right < 0.5)


# =========================
# EVITACIÓN
# =========================
def wall_bias(sonar):
    front = min(sonar[3:6])
    left = min(sonar[6:10])
    right = min(sonar[0:3])

    if front < 0.25:
        return -2.0, 2.0

    bias = np.clip(right - left, -0.4, 0.4)
    turn = 1.5 * bias
    base = 1.5

    return base - turn, base + turn


# =========================
# SEGUIMIENTO BOLA (MEJORADO)
# =========================
def follow_ball(ball, width):
    global last_cx, last_seen_time, smooth_cx

    center = width // 2

    if ball:
        cx, w, area = ball

        # suavizado
        if smooth_cx is None:
            smooth_cx = cx
        else:
            smooth_cx = int(alpha * cx + (1 - alpha) * smooth_cx)

        last_cx = smooth_cx
        last_seen_time = 0

        error = (center - smooth_cx) / center
        if abs(error) < 0.08:
            error = 0

        # más pegado a la bola
        target_area = 45000
        dist_error = (target_area - area) / target_area

        base = 2.8 + 4.0 * dist_error
        base = np.clip(base, 1.5, 5.5)

        turn = 1.2 * error
        turn = np.clip(turn, -1.5, 1.5)

        base *= (1 - 0.4 * abs(error))

        return base - turn, base + turn

    # =========================
    # MEMORIA CUANDO SE PIERDE
    # =========================
    last_seen_time += 1

    if last_cx is None:
        return 1.0, -1.0

    if last_seen_time < 20:
        if last_cx < center:
            return 0.8, 2.6
        else:
            return 2.6, 0.8

    return 1.0, -1.0


# =========================
# CONTROL PRINCIPAL
# =========================
def controller(sonar, ball, width):
    global corner_mode, corner_dir

    front = min(sonar[3:6])
    left_s = min(sonar[6:10])
    right_s = min(sonar[0:3])

    # -------------------------
    # ESQUINA
    # -------------------------
    if corner_mode:
        if front > 0.7:
            corner_mode = False
        else:
            corner_cmd = (-0.5 * corner_dir, 1.8 * corner_dir)

            if ball is not None:
                ball_cmd = follow_ball(ball, width)
                return (
                    0.8 * ball_cmd[0] + 0.2 * corner_cmd[0],
                    0.8 * ball_cmd[1] + 0.2 * corner_cmd[1],
                )
            return corner_cmd

    if is_corner(front, left_s, right_s):
        corner_mode = True
        corner_dir = 1 if right_s > left_s else -1
        return (-0.5 * corner_dir, 1.8 * corner_dir)

    # -------------------------
    # EMERGENCIA
    # -------------------------
    if front < 0.25:
        return -2.0, 2.0

    # -------------------------
    # BOLA + PAREDES
    # -------------------------
    wall = wall_bias(sonar)

    if ball is None:
        return follow_ball(None, width)

    ball_cmd = follow_ball(ball, width)

    return (
        0.7 * ball_cmd[0] + 0.3 * wall[0],
        0.7 * ball_cmd[1] + 0.3 * wall[1],
    )


# =========================
# MAIN
# =========================
def main():
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', use_camera=True)

    coppelia.start_simulation()

    try:
        while coppelia.is_running():

            img = robot.get_image()
            ball = detect_ball(img)
            sonar = robot.get_sonar()

            left, right = controller(sonar, ball, img.shape[1])
            robot.set_speed(left, right)

            if ball:
                cx, w, area = ball
                cv2.line(img, (cx, 0), (cx, img.shape[0]), (0,255,0), 2)

            cv2.imshow("camera", img)
            cv2.waitKey(1)

    finally:
        coppelia.stop_simulation()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()