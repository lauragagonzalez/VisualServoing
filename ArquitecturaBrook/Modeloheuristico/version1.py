

"""
Búsqueda: heurístico reactivo (giro hasta detectar bola)
Seguimiento: controlador proporcional (P) heurístico
Evitación: control heuristico
Arquitectura: ES reactiva basada en comportamientos (Brooks)
"""


import cv2
import numpy as np
import robotica

# DETECCIÓN DE BOLA
def detect_ball(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 120, 70])
    upper1 = np.array([10, 255, 255])

    lower2 = np.array([170, 120, 70])
    upper2 = np.array([180, 255, 255])

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2)
    )

    area = cv2.countNonZero(mask)

    if area < 50:
        return None

    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    return cx, img.shape[1], area


# EVITACIÓN DE OBSTÁCULOS 
def avoid_obstacles(sonar):
    front = min(sonar[3:6])
    left = min(sonar[6:10])
    right = min(sonar[0:3])
    if front < 0.35:
        return -2.0, 2.0

    if left < 0.25:
        return 1.5, 2.2

    if right < 0.25:
        return 2.2, 1.5

    return None



# SEGUIMIENTO DE BOLA
last_cx = None
def follow_ball(ball, width):
    global last_cx

    if ball:
        cx, w, area = ball
        last_cx = cx

        center = width // 2
        error = center - cx

        # 📏 velocidad según distancia
        if area < 1000:
            base = 2.5
        elif area < 3000:
            base = 1.8
        else:
            base = 1.2

        # 🎯 dirección
        if abs(error) < 20:
            return base, base
        elif error > 0:
            return base * 0.6, base * 1.2
        else:
            return base * 1.2, base * 0.6

    # 🔍 búsqueda si se pierde
    if last_cx is not None:
        if last_cx < width / 2:
            return 0.8, 1.6
        else:
            return 1.6, 0.8

    return 0.7, -0.7


def main():
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', use_camera=True)

    coppelia.start_simulation()

    try:
        while coppelia.is_running():
            img = robot.get_image()
            ball = detect_ball(img)
            if ball:
                cx, w, area = ball
            else:
                print("NO DETECTA BOLA")

            sonar = robot.get_sonar()

            # EVITACIÓN 
            avoid = avoid_obstacles(sonar)
            if avoid:
                robot.set_speed(*avoid)
            else:
                # SEGUIMIENTO BOLA
                left, right = follow_ball(ball, img.shape[1])
                robot.set_speed(left, right)
            if ball:
                cx, w, area = ball
                cv2.line(img, (cx, 0), (cx, img.shape[0]), (0, 255, 0), 2)

            cv2.line(img, (img.shape[1] // 2, 0),
                     (img.shape[1] // 2, img.shape[0]),
                     (255, 255, 0), 1)

            cv2.imshow("camera", img)
            cv2.waitKey(1)

    finally:
        coppelia.stop_simulation()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()