import cv2
import numpy as np
import robotica


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



last_cx = None
last_seen_time = 0
smooth_cx = None
alpha = 0.6

def is_corner(front, left, right):
    return front < 0.45 and (left < 0.5 or right < 0.5)


def safety_layer(front):
    if front < 0.25:
        return -2.0, 2.0
    return None



def corner_layer(front, left, right):
    if is_corner(front, left, right):
        if right > left:
            return -0.6, 2.0
        else:
            return 2.0, -0.6
    return None


def avoid_layer(front, left, right):
    if front < 0.5:
        if right > left:
            return -1.2, 2.2
        else:
            return 2.2, -1.2
    return None



def follow_ball(ball, width):
    global last_cx, last_seen_time, smooth_cx

    center = width // 2

    if ball:
        cx, w, area = ball
        if smooth_cx is None:
            smooth_cx = cx
        else:
            smooth_cx = int(alpha * cx + (1 - alpha) * smooth_cx)

        last_cx = smooth_cx
        last_seen_time = 0

        error = (center - smooth_cx) / center

        if abs(error) < 0.15:
            error = 0

        turn = 1.0 * error

        if area > 40000:
            turn *= 0.5

        turn = np.clip(turn, -0.8, 0.8)


        target_area = 45000
        dist_error = (target_area - area) / target_area

        if abs(dist_error) < 0.12:
            dist_error = 0

        base = 2.2 + 2.0 * dist_error

        # evitar pegarse
        if area > 65000:
            base *= 0.3

        if area < 25000:
            base *= 1.4

        base = np.clip(base, 0.2, 3.5)
        base *= (1 - 0.3 * abs(error))

        return base - turn, base + turn

    last_seen_time += 1

    if last_cx is None:
        return None

    if last_seen_time < 20:
        if last_cx < center:
            return 0.8, 2.4
        else:
            return 2.4, 0.8

    return None


# ARQUITECTURA DE CONNELL
def controller(sonar, ball, width):
    front = min(sonar[3:6])
    left = min(sonar[6:10])
    right = min(sonar[0:3])

    cmd = safety_layer(front)
    if cmd is not None:
        return cmd

    corner_cmd = corner_layer(front, left, right)
    avoid_cmd = avoid_layer(front, left, right)
    follow_cmd = follow_ball(ball, width)

    if corner_cmd is not None:
        if follow_cmd is not None:
            return (
                0.6 * follow_cmd[0] + 0.4 * corner_cmd[0],
                0.6 * follow_cmd[1] + 0.4 * corner_cmd[1],
            )
        return corner_cmd

    if avoid_cmd is not None:
        if follow_cmd is not None:
            return (
                0.7 * follow_cmd[0] + 0.3 * avoid_cmd[0],
                0.7 * follow_cmd[1] + 0.3 * avoid_cmd[1],
            )
        return avoid_cmd

    if follow_cmd is not None:
        return follow_cmd

    return 1.0, -1.0



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
                cv2.line(img, (cx, 0), (cx, img.shape[0]), (0, 255, 0), 2)

            cv2.imshow("camera", img)
            cv2.waitKey(1)

    finally:
        coppelia.stop_simulation()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
