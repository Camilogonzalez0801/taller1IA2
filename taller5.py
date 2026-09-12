"""
Inteligencia Artificial II - Sesion 5: Gradientes Espaciales y Deteccion de Bordes
Solucion del Taller de Laboratorio: Inspector de Bordes
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Cargar una imagen con formas geometricas claras y texturas
# ============================================================
ruta_imagen = "escena.jpg"
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

if imagen is None:
    print(f"[Aviso] No se encontro '{ruta_imagen}'. Generando una imagen de "
          f"ejemplo (formas geometricas + una zona con textura) para poder "
          f"correr el resto del codigo.")
    alto, ancho = 300, 300
    imagen = np.full((alto, ancho), 200, dtype=np.uint8)  # fondo claro

    # Forma 1: cuadrado oscuro (borde limpio)
    imagen[40:140, 30:130] = 30

    # Forma 2: circulo gris medio (borde limpio, curvo)
    yy, xx = np.ogrid[:alto, :ancho]
    mascara_circulo = (xx - 210) ** 2 + (yy - 90) ** 2 <= 55 ** 2
    imagen[mascara_circulo] = 100

    # Zona con "textura compleja": rayas finas alternadas (simula ladrillos/fachada)
    textura = imagen[180:280, 40:260]
    for col in range(textura.shape[1]):
        if col % 6 < 3:
            textura[:, col] = 160
        else:
            textura[:, col] = 130
    imagen[180:280, 40:260] = textura

    cv2.imwrite("0_escena_original.png", imagen)

# ============================================================
# 2. Generar 3 variables separadas: Sobel X, Sobel Y y Canny
# ============================================================
sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)

sobel_x_abs = cv2.convertScaleAbs(sobel_x)   # para poder visualizarlo (0-255)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

bordes_canny = cv2.Canny(imagen, 50, 150)    # umbrales "normales" de ejemplo

# ============================================================
# 3. Panel de visualizacion para comparar los tres
# ============================================================
fig, ejes = plt.subplots(1, 4, figsize=(16, 4))
titulos = ["Original", "Sobel X (bordes verticales)", "Sobel Y (bordes horizontales)", "Canny (50, 150)"]
imagenes = [imagen, sobel_x_abs, sobel_y_abs, bordes_canny]

for ax, img, titulo in zip(ejes, imagenes, titulos):
    ax.imshow(img, cmap="gray")
    ax.set_title(titulo, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("1_panel_sobel_canny.png", dpi=150)
print("Panel guardado como '1_panel_sobel_canny.png'")

# ============================================================
# 4. Experimentacion: variar los umbrales de Canny
# ============================================================
canny_bajo = cv2.Canny(imagen, 10, 50)      # umbrales muy bajos -> mas sensible
canny_alto = cv2.Canny(imagen, 200, 250)    # umbrales muy altos -> mas estricto

fig2, ejes2 = plt.subplots(1, 3, figsize=(12, 4))
titulos2 = ["Canny (10, 50)\n-> muy sensible, capta hasta la textura",
            "Canny (50, 150)\n-> equilibrado",
            "Canny (200, 250)\n-> muy estricto, solo bordes fuertes"]
imagenes2 = [canny_bajo, bordes_canny, canny_alto]

for ax, img, titulo in zip(ejes2, imagenes2, titulos2):
    ax.imshow(img, cmap="gray")
    ax.set_title(titulo, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("2_experimentacion_umbrales.png", dpi=150)
print("Comparacion de umbrales guardada como '2_experimentacion_umbrales.png'")

# ============================================================
# Analisis numerico: cuantos pixeles de borde detecta cada umbral
# ============================================================
print("\nPixeles de borde detectados (blancos) segun el umbral de Canny:")
print("  Umbral (10, 50)   :", cv2.countNonZero(canny_bajo))
print("  Umbral (50, 150)  :", cv2.countNonZero(bordes_canny))
print("  Umbral (200, 250) :", cv2.countNonZero(canny_alto))