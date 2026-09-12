"""
Inteligencia Artificial II - Sesion 4: Convolucion y Filtrado
Solucion del Taller de Laboratorio: Estrategias de Suavizado
"""

import cv2
import numpy as np

# ============================================================
# 1. Crear una imagen con mucho ruido de Sal y Pimienta
# ============================================================
rng = np.random.default_rng(11)
alto, ancho = 300, 300

# Imagen base: un cuadrado gris claro sobre fondo gris oscuro
# (para poder ver despues si los filtros respetan o no ese borde)
imagen = np.full((alto, ancho), 90, dtype=np.uint8)
imagen[80:220, 80:220] = 180

# Ruido de sal (pixeles blancos) y pimienta (pixeles negros) dispersos al azar
porcentaje_ruido = 0.05  # 5% de los pixeles
n_ruido = int(alto * ancho * porcentaje_ruido)

coords_sal = (rng.integers(0, alto, n_ruido // 2), rng.integers(0, ancho, n_ruido // 2))
imagen[coords_sal] = 255

coords_pimienta = (rng.integers(0, alto, n_ruido // 2), rng.integers(0, ancho, n_ruido // 2))
imagen[coords_pimienta] = 0

cv2.imwrite("0_original_con_ruido.png", imagen)

# ============================================================
# 2. Aplicar los 3 filtros con un kernel agresivo (7x7)
# ============================================================
blur_media = cv2.blur(imagen, (7, 7))
blur_gauss = cv2.GaussianBlur(imagen, (7, 7), 0)
blur_mediana = cv2.medianBlur(imagen, 7)

cv2.imwrite("1_filtro_media.png", blur_media)
cv2.imwrite("2_filtro_gaussiano.png", blur_gauss)
cv2.imwrite("3_filtro_mediana.png", blur_mediana)

# ============================================================
# 3. Mostrar los resultados (en un IDE normal, con ventanas)
# ============================================================
# cv2.imshow("Original con ruido", imagen)
# cv2.imshow("Filtro de Media 7x7", blur_media)
# cv2.imshow("Filtro Gaussiano 7x7", blur_gauss)
# cv2.imshow("Filtro de Mediana 7x7", blur_mediana)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ============================================================
# 4. Analisis critico: verificar numericamente que la mediana
#    "ignora" los extremos y la media los deja como mancha
# ============================================================
# Buscamos un pixel de ruido (sal, valor 255) y miramos su vecindad 7x7
fy, fx = coords_sal[0][0], coords_sal[1][0]
y0, y1 = max(fy - 3, 0), min(fy + 4, alto)
x0, x1 = max(fx - 3, 0), min(fx + 4, ancho)
vecindad_original = imagen[y0:y1, x0:x1]

print(f"Pixel de ruido de ejemplo en ({fy},{fx}), valor original: {imagen[fy, fx]}")
print("\nSu vecindad 7x7 en la imagen original (con ruido):")
print(vecindad_original)

print(f"\nValor tras Filtro de Media   : {blur_media[fy, fx]}  (queda una 'mancha' gris, mezcla de todo)")
print(f"Valor tras Filtro Gaussiano  : {blur_gauss[fy, fx]}  (mancha mas suave, pero sigue mezclando)")
print(f"Valor tras Filtro de Mediana : {blur_mediana[fy, fx]}  (recupera el valor real del fondo, sin mancha)")