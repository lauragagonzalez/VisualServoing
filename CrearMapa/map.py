import robotica
import numpy as np
import time
import matplotlib.pyplot as plt


RESOLUTION = 0.02
MAP_SIZE = 1000
MAP_OFFSET = MAP_SIZE // 2

# 0 = libre
# 0.5 = desconocido
# 1 = ocupado
grid_map = np.ones((MAP_SIZE, MAP_SIZE)) * 0.5
estado = "INICIO"


def cerca(x, umbral=0.5):
    return max(0, min(1, (umbral - x) / umbral))


def medio(x, centro=0.7, ancho=0.4):
    return max(0, 1 - abs(x - centro) / ancho)


def lejos(x, umbral=1.0):
    return max(0, min(1, (x - umbral) / (2.0 - umbral)))


def defuzz(reglas):

    num = sum(p * v for p, v in reglas)
    den = sum(p for p, v in reglas)

    if den == 0:
        return 0

    return num / den



def control(readings):

    global estado

    front = min(readings[3], readings[4])

    front_r = readings[5]

    right = readings[6]

    right_diag = readings[7]

    back_right = readings[0]
    if min(front, front_r) < 0.22:

        estado = "GIRO"

        return -0.8, 1.2

    if estado == "INICIO":
        estado = "SEGUIR"

    front_avg = (front + front_r) / 2

    vel = defuzz([
        (lejos(front_avg, 1.2), 1.5),
        (medio(front_avg, 0.9, 0.3), 1.0),
        (cerca(front_avg, 0.5), 0.55),
    ])

    right_mix = (
        0.45 * right +
        0.35 * right_diag +
        0.20 * front_r
    )

    target = 0.50

    error = right_mix - target

    # esquinas suaves
    corner = (1 - lejos(front_r, 0.8))

    error += corner * 0.22

    # suavizado
    error = (
        0.7 * error +
        0.3 * (right - back_right)
    )

    if abs(error) < 0.03:
        error = 0

    giro = error * 1.2

    giro = max(min(giro, 0.45), -0.45)

    # velocidad adaptativa
    vel = vel * (1 - abs(giro) * 0.25)

    vel = max(min(vel, 1.8), 0.5)

    lspeed = vel + giro
    rspeed = vel - giro

    return lspeed, rspeed



def bresenham(x0, y0, x1, y1):

    points = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    while True:

        points.append((x0, y0))

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x0 += sx

        if e2 < dx:
            err += dx
            y0 += sy

    return points


def update_map(rx, ry, ox, oy):

    gx = int(rx / RESOLUTION) + MAP_OFFSET
    gy = int(ry / RESOLUTION) + MAP_OFFSET

    for i in range(len(ox)):

        ogx = int(ox[i] / RESOLUTION) + MAP_OFFSET
        ogy = int(oy[i] / RESOLUTION) + MAP_OFFSET

        if not (2 <= ogx < MAP_SIZE - 2):
            continue

        if not (2 <= ogy < MAP_SIZE - 2):
            continue

        ray = bresenham(gx, gy, ogx, ogy)

        for x, y in ray[:-1]:

            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:

                # NO borrar obstáculos fuertes
                if grid_map[x, y] < 0.9:

                    grid_map[x, y] -= 0.04

                    grid_map[x, y] = max(
                        0.0,
                        grid_map[x, y]
                    )
        

        for dx in range(0,2):
            for dy in range(0, 2):

                nx = ogx + dx
                ny = ogy + dy

                if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:

                    grid_map[nx, ny] += 0.12

                    grid_map[nx, ny] = min(
                        1.0,
                        grid_map[nx, ny]
                    )



def draw_map(rx, ry):

    plt.clf()

    plt.imshow(
        grid_map.T,
        cmap="binary",
        origin="lower",
        vmin=0,
        vmax=1
    )

    gx = int(rx / RESOLUTION) + MAP_OFFSET
    gy = int(ry / RESOLUTION) + MAP_OFFSET

    plt.plot(
        gx,
        gy,
        "ro",
        markersize=4
    )

    plt.xlim(gx - 180, gx + 180)
    plt.ylim(gy - 180, gy + 180)

    plt.grid(True, alpha=0.15)

    plt.pause(0.001)


def main():

    coppelia = robotica.Coppelia()

    robot = robotica.P3DX(
        coppelia.sim,
        "PioneerP3DX",
        use_lidar=True
    )

    coppelia.start_simulation()

    time.sleep(1)

    print("Simulación iniciada")

    robot_handle = coppelia.sim.getObject(
        "/PioneerP3DX"
    )

    plt.figure(figsize=(8, 8))

    plt.ion()

    FOV = 4.18879

    try:

        while coppelia.is_running():

            lidar = np.array(
                robot.get_lidar()
            )

            if len(lidar) == 0:
                continue

            angles = np.linspace(
                -FOV / 2,
                 FOV / 2,
                 len(lidar)
            )

            valid = (
                (lidar > 0.05) &
                (lidar < 4.0)
            )

            lidar = lidar[valid]
            angles = angles[valid]

            pos = coppelia.sim.getObjectPosition(
                robot_handle,
                -1
            )

            ori = coppelia.sim.getObjectOrientation(
                robot_handle,
                -1
            )

            rx = pos[0]
            ry = pos[1]
            yaw = ori[2]

            ox = rx + np.cos(
                angles + yaw
            ) * lidar

            oy = ry + np.sin(
                angles + yaw
            ) * lidar

            update_map(
                rx,
                ry,
                ox,
                oy
            )

            sonar = robot.get_sonar()

            ls, rs = control(sonar)

            robot.set_speed(ls, rs)

            draw_map(rx, ry)

            time.sleep(0.05)

    finally:

        np.save(
            "mapa.npy",
            grid_map
        )

        plt.figure(figsize=(10, 10))

        plt.imshow(
            grid_map.T,
            cmap="binary",
            origin="lower",
            vmin=0,
            vmax=1
        )

        plt.title(
            "Occupancy Grid Map"
        )

        plt.savefig(
            "mapa.png",
            dpi=300
        )

        print("mapa.npy guardado")
        print("mapa.png guardado")

        robot.set_speed(0, 0)

        coppelia.stop_simulation()

        print("Fin")


if __name__ == "__main__":
    main()