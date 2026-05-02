import numpy as np
import cv2

grid = np.load("grid_map.npy")

# normalizar a imagen
img = (grid * 255).astype(np.uint8)

# hacerlo grande para verlo bien
img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_NEAREST)

cv2.imshow("Mapa del robot", img)
cv2.waitKey(0)
cv2.destroyAllWindows()