"""
Inteligencia Artificial II - Sesion 6: Extraccion de Caracteristicas y Contornos
Solucion del Taller de Laboratorio: Clasificador de Formas (Proyecto Integrador Modulo 1)

Pipeline completo: Grises -> Umbralizacion -> Limpieza morfologica -> Contornos
"""

import cv2
import numpy as np

# ============================================================
# 1. Cargar (o generar) una imagen con varios objetos de
#    distintos tamaños sobre un fondo uniforme
# ============================================================
ruta_imagen = "monedas.jpg"
imagen_color = cv2.imread(ruta_imagen)

if imagen_color is None:
    print(f"[Aviso] No se encontro '{ruta_imagen}'. Generando una imagen de "
          f"ejemplo (varias 'monedas' circulares de distintos tamaños sobre "
          f"un fondo uniforme) para poder correr el resto del codigo.")
    alto, ancho = 400, 600
    imagen_color = np.full((alto, ancho, 3), 230, dtype=np.uint8)  # fondo claro uniforme

    # (centro_x, centro_y, radio) -> simulan monedas de distinto tamaño
    monedas = [(90, 90, 30), (220, 100, 45), (370, 90, 60),
               (520, 100, 22), (150, 280, 55), (400, 280, 38)]
    for cx, cy, r in monedas:
        cv2.circle(imagen_color, (cx, cy), r, (120, 120, 120), -1)

    cv2.imwrite("0_monedas_original.png", imagen_color)

# ============================================================
# 2. Pipeline: Grises -> Umbralizacion -> Limpieza morfologica
# ============================================================
imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

# Umbralizacion automatica con Otsu (el fondo es mas claro que los objetos,
# asi que invertimos para que los objetos queden en blanco)
_, imagen_binaria = cv2.threshold(
    imagen_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Limpieza morfologica: Apertura para quitar ruido de fondo,
# Cierre para tapar huequitos dentro de los objetos
kernel = np.ones((5, 5), np.uint8)
imagen_limpia = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)
imagen_limpia = cv2.morphologyEx(imagen_limpia, cv2.MORPH_CLOSE, kernel)

cv2.imwrite("1_binaria_limpia.png", imagen_limpia)

# ============================================================
# 3. Deteccion de Contornos
# ============================================================
contornos, jerarquia = cv2.findContours(
    imagen_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# ============================================================
# 4. Area de cada objeto + logica empresarial (Bounding Box
#    Azul si es grande, Rojo si es pequeño)
# ============================================================
imagen_resultado = imagen_color.copy()

areas = []
for cnt in contornos:
    area = cv2.contourArea(cnt)
    if area > 500:  # filtramos ruido/objetos minusculos
        areas.append(area)

# Umbral X de clasificacion: aqui se calcula automaticamente como el
# promedio de las areas encontradas, para separar "grandes" de "pequeños"
# sin tener que adivinar un numero fijo a mano.
X = sum(areas) / len(areas) if areas else 0
print(f"Umbral de clasificacion (X) calculado como el promedio de areas: {X:.1f} px\n")

contador = 1
for cnt in contornos:
    area = cv2.contourArea(cnt)
    if area > 500:
        x, y, w, h = cv2.boundingRect(cnt)

        if area > X:
            color = (255, 0, 0)   # Azul (BGR) -> objeto grande
            etiqueta = "GRANDE"
        else:
            color = (0, 0, 255)   # Rojo (BGR) -> objeto pequeño
            etiqueta = "pequeño"

        cv2.rectangle(imagen_resultado, (x, y), (x + w, y + h), color, 3)

        # Centroide con los momentos, como en el ejemplo de la guia
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(imagen_resultado, (cx, cy), 4, (0, 0, 0), -1)

        print(f"Objeto {contador}: area = {area:.1f} px  ->  {etiqueta}  "
              f"(Bounding Box: x={x}, y={y}, w={w}, h={h})")
        contador += 1

cv2.imwrite("2_resultado_clasificado.png", imagen_resultado)
print("\nImagenes guardadas: 0_monedas_original.png, 1_binaria_limpia.png, 2_resultado_clasificado.png")

# cv2.imshow("Original", imagen_color)
# cv2.imshow("Binaria limpia", imagen_limpia)
# cv2.imshow("Clasificacion final", imagen_resultado)
# cv2.waitKey(0)
# cv2.destroyAllWindows()