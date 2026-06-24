import cv2
import numpy as np

# Cargar imágenes
img1_color = cv2.imread("data/align.jpeg")  # Imagen desalineada
img2_color = cv2.imread("data/ref.jpeg")    # Imagen de referencia

# Verificar carga
if img1_color is None:
    print("Error: No se pudo cargar data/align.jpeg")
    exit()

if img2_color is None:
    print("Error: No se pudo cargar data/ref.jpeg")
    exit()

# Convertir a escala de grises
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

height, width = img2.shape

# Crear detector ORB
orb_detector = cv2.ORB_create(5000)

# Detectar puntos clave y descriptores
kp1, d1 = orb_detector.detectAndCompute(img1, None)
kp2, d2 = orb_detector.detectAndCompute(img2, None)

# Verificar que se encontraron características
if d1 is None or d2 is None:
    print("Error: No se encontraron suficientes características ORB.")
    exit()

# Emparejamiento
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Convertir a lista para evitar error de OpenCV 4.13
matches = list(matcher.match(d1, d2))

# Verificar matches
if len(matches) < 4:
    print("Error: No hay suficientes coincidencias.")
    exit()

# Ordenar coincidencias
matches = sorted(matches, key=lambda x: x.distance)

# Tomar el 90% de las mejores
matches = matches[:int(len(matches) * 0.9)]

no_of_matches = len(matches)

print("Número de matches:", no_of_matches)

# Crear matrices de puntos
p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))

for i in range(no_of_matches):
    p1[i, :] = kp1[matches[i].queryIdx].pt
    p2[i, :] = kp2[matches[i].trainIdx].pt

# Calcular homografía
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

if homography is None:
    print("Error: No se pudo calcular la homografía.")
    exit()

# Corregir perspectiva
transformed_img = cv2.warpPerspective(
    img1_color,
    homography,
    (width, height)
)

# Guardar resultado
cv2.imwrite("data/output.jpg", transformed_img)

print("Imagen corregida guardada en data/output.jpg")

# Mostrar imágenes
cv2.imshow("Referencia", img2_color)
cv2.imshow("Desalineada", img1_color)
cv2.imshow("Corregida", transformed_img)

cv2.waitKey(0)
cv2.destroyAllWindows()