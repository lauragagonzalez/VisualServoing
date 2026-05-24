# arquitectura reactiva por comportamientos
# busqueda: giro si no hay bola
# seguimiento: control proporcional
# evitacion: prioridad maxima

import cv2
import numpy as np
import robotica


def detectar_bola(imagen):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # el rojo en HSV se parte en dos rangos, lo vi en stackoverflow
    lim_inf1 = np.array([0, 120, 70])
    lim_sup1 = np.array([10, 255, 255])
    lim_inf2 = np.array([170, 120, 70])
    lim_sup2 = np.array([180, 255, 255])

    mascara = cv2.bitwise_or(
        cv2.inRange(hsv, lim_inf1, lim_sup1),
        cv2.inRange(hsv, lim_inf2, lim_sup2)
    )

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        return None

    # me quedo con el mas grande
    c = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < 20:  # descarto ruido pequeño
        return None

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy, area


def evitar_obstaculos(sonar):
    frente = min(sonar[3:6])
    izq    = min(sonar[6:10])
    der    = min(sonar[0:3])

    if frente < 0.35:
        return -2.0, 2.0   # giro en sitio

    if izq < 0.25:
        return 1.5, 2.2    # me alejo de la pared izquierda

    if der < 0.25:
        return 2.2, 1.5

    return None


ultima_x = None   # para saber hacia donde girar si pierdo la bola

def seguir_bola(bola, ancho):
    global ultima_x

    if bola:
        cx, cy, area = bola
        ultima_x = cx

        error = (ancho // 2) - cx

        # estos valores los fui probando hasta que me funciono
        if area < 1000:
            base = 2.5
        elif area < 3000:
            base = 1.8
        else:
            base = 1.2 

        if abs(error) < 20:
            return base, base
        elif error > 0:
            return base * 0.6, base * 1.2
        else:
            return base * 1.2, base * 0.6

    # bola no visible, intento girar hacia donde la vi por ultima vez
    if ultima_x is not None:
        if ultima_x < ancho / 2:
            return 0.8, 1.6
        else:
            return 1.6, 0.8
    return None


def buscar_bola(bola):
    # si nunca la vi, giro sobre mi mismo buscando
    if bola is not None:
        return None
    return 0.7, -0.7


def prioridad(evitar, seguir, buscar):
    """
    Orden de prioridad:
    1. evitar (más fuerte)
    2. seguir
    3. buscar
    """
    if evitar is not None:
        return evitar

    if seguir is not None:
        return seguir

    return buscar

def main():
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', use_camera=True)

    coppelia.start_simulation()

    try:
        while coppelia.is_running():

            imagen = robot.get_image()
            sonar = robot.get_sonar()
            bola = detectar_bola(imagen)

            ev = evitar_obstaculos(sonar)
            se = seguir_bola(bola, imagen.shape[1])
            bu = buscar_bola(bola)

            izq, der = prioridad(ev, se, bu)

            robot.set_speed(izq, der)

            if bola:
                cx, cy, area = bola
                cv2.circle(imagen, (cx, cy), 5, (0, 255, 0), -1)
                cv2.line(imagen, (cx, 0), (cx, imagen.shape[0]), (0, 255, 0), 2)

            cv2.imshow("Camara", imagen)
            cv2.waitKey(1)

    finally:
        coppelia.stop_simulation()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()