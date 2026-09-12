"""
Inteligencia Artificial II - Sesion 3: Segmentacion
Solucion del Taller de Laboratorio Final: Limpiando la Vision (morfologia matematica)
"""

import cv2
import numpy as np

# ============================================================
# 1. Cargar una imagen en escala de grises con poco contraste
# ============================================================
ruta_imagen = "documento.jpg"
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

if imagen is None:
    print(f"[Aviso] No se encontro '{ruta_imagen}'. Generando una imagen de "
          f"ejemplo (un objeto claro sobre fondo oscuro, con poco contraste "
          f"y algo de ruido) para poder correr el resto del codigo.")
    rng = np.random.default_rng(7)
    alto, ancho = 300, 300
    # Fondo oscuro con algo de variacion (poco contraste real)
    imagen = np.clip(70 + rng.normal(0, 12, (alto, ancho)), 0, 255).astype(np.uint8)
    # "Objeto" de interes: un circulo mas claro en el centro
    yy, xx = np.ogrid[:alto, :ancho]
    mascara_circulo = (xx - 150) ** 2 + (yy - 150) ** 2 <= 70 ** 2
    imagen[mascara_circulo] = np.clip(
        150 + rng.normal(0, 12, mascara_circulo.sum()), 0, 255
    ).astype(np.uint8)
    cv2.imwrite("imagen_original.png", imagen)

# ============================================================
# 2. Umbralizacion estatica (a proposito, con un valor que deja ruido)
# ============================================================
T = 110
_, imagen_binaria = cv2.threshold(imagen, T, 255, cv2.THRESH_BINARY)
cv2.imwrite("1_binarizada_con_ruido.png", imagen_binaria)
# cv2.imshow("Binarizada (con ruido)", imagen_binaria)

# ============================================================
# 3. Elemento Estructurante 3x3
# ============================================================
kernel = np.ones((3, 3), np.uint8)

# ============================================================
# 4. Apertura (Erosion + Dilatacion): limpia el fondo sin encoger
#    permanentemente el objeto principal
# ============================================================
erosionada = cv2.erode(imagen_binaria, kernel, iterations=1)
apertura = cv2.dilate(erosionada, kernel, iterations=1)
cv2.imwrite("2_apertura.png", apertura)
# cv2.imshow("Apertura", apertura)

# ============================================================
# 5. Cierre (Dilatacion + Erosion): rellena huecos internos del objeto
# ============================================================
dilatada = cv2.dilate(imagen_binaria, kernel, iterations=1)
cierre = cv2.erode(dilatada, kernel, iterations=1)
cv2.imwrite("3_cierre.png", cierre)
# cv2.imshow("Cierre", cierre)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ============================================================
# 6. Comparacion: contar pixeles blancos "sueltos" (ruido) que sobrevivieron
# ============================================================
ruido_original = cv2.countNonZero(imagen_binaria)
ruido_apertura = cv2.countNonZero(apertura)
ruido_cierre = cv2.countNonZero(cierre)

print("Pixeles blancos en la binarizada original :", ruido_original)
print("Pixeles blancos despues de la Apertura    :", ruido_apertura)
print("Pixeles blancos despues del Cierre        :", ruido_cierre)
print("\nImagenes guardadas: 1_binarizada_con_ruido.png, 2_apertura.png, 3_cierre.png")