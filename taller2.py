"""
Inteligencia Artificial II - Sesion 2: Tensor de Color y Analisis Estadistico
Solucion de los talleres de laboratorio (practicos).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# TALLER DE LABORATORIO 1: TRANSFORMACION DE ESPACIOS
# ============================================================
print("=" * 60)
print("TALLER DE LABORATORIO 1: TRANSFORMACION DE ESPACIOS")
print("=" * 60)

# 1. Pixel BGR de prueba: amarillo intenso puro
#    OJO: el orden es RGB (Azul, Verde, Rojo), no RGB
pixel = np.array([0, 255, 255])  # B=0, G=255, R=255
print("\nPixel BGR de prueba (amarillo puro):", pixel)

# 2. Calculo matematico manual con la formula ponderada
#    Y = 0.299*R + 0.587*G + 0.114*B
B, G, R = pixel[0], pixel[1], pixel[2]
Y = 0.299 * R + 0.587 * G + 0.114 * B

# 3. Resultado
print("Valor en escala de grises (calculo manual): Y =", Y)
print("Redondeado a entero (0-255):", round(Y))

# 4. Verificacion con la funcion optimizada de OpenCV
#    Creamos una "imagen" de 1x1 pixel para poder usar cv2.cvtColor
imagen_1px = np.uint8([[pixel]])  # forma (1,1,3)
gris_cv2 = cv2.cvtColor(imagen_1px, cv2.COLOR_BGR2GRAY)
print("Valor en escala de grises (cv2.cvtColor):", gris_cv2[0, 0])


# ============================================================
# TALLER DE LABORATORIO 2: ANALISIS ESTADISTICO (HISTOGRAMAS)
# ============================================================
print("\n" + "=" * 60)
print("TALLER DE LABORATORIO 2: ANALISIS ESTADISTICO")
print("=" * 60)

# 1. Cargar una imagen RGB (aqui se carga desde archivo; cambien la ruta
#    por la de su propia foto, por ejemplo: 'mi_foto.jpg')
ruta_imagen = "mi_foto.jpg"
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print(f"\n[Aviso] No se encontro '{ruta_imagen}'. Generando una imagen de "
          f"ejemplo para poder correr el resto del codigo.")
    # Imagen sintetica solo para que el script sea ejecutable de una vez;
    # en la entrega real deben reemplazarla por una foto real (paso 1 del taller).
    rng = np.random.default_rng(42)
    imagen = np.zeros((300, 300, 3), dtype=np.uint8)
    imagen[:, :, 0] = np.clip(60 + rng.normal(0, 15, (300, 300)), 0, 255)   # Azul bajo, con ruido
    x = np.linspace(40, 210, 300)
    base_verde = np.tile(x, (300, 1))
    imagen[:, :, 1] = np.clip(base_verde + rng.normal(0, 12, (300, 300)), 0, 255)  # Verde en degrade
    imagen[:, :, 2] = np.clip(120 + rng.normal(0, 25, (300, 300)), 0, 255)  # Rojo medio, con ruido

# 2. Separar la imagen en sus 3 canales (recordar: OpenCV usa orden BGR)
canal_azul = imagen[:, :, 0]
canal_verde = imagen[:, :, 1]
canal_rojo = imagen[:, :, 2]

# 3. Calcular el histograma de cada canal por separado
hist_azul = cv2.calcHist([imagen], [0], None, [256], [0, 256])
hist_verde = cv2.calcHist([imagen], [1], None, [256], [0, 256])
hist_rojo = cv2.calcHist([imagen], [2], None, [256], [0, 256])

# 4. Graficar los tres histogramas superpuestos
plt.figure(figsize=(8, 5))
plt.plot(hist_azul, color="blue", label="Canal Azul (B)")
plt.plot(hist_verde, color="green", label="Canal Verde (G)")
plt.plot(hist_rojo, color="red", label="Canal Rojo (R)")
plt.title("Histograma de color por canal")
plt.xlabel("Valor del pixel (0-255)")
plt.ylabel("Frecuencia (cantidad de pixeles)")
plt.legend()
plt.tight_layout()
plt.savefig("histograma_canales.png", dpi=150)
print("\nHistograma guardado como 'histograma_canales.png'")
plt.show()

# 5. Analisis simple del color dominante: comparamos el promedio de cada canal
promedio_azul = canal_azul.mean()
promedio_verde = canal_verde.mean()
promedio_rojo = canal_rojo.mean()

print("\nPromedio de intensidad por canal:")
print(f"  Azul : {promedio_azul:.2f}")
print(f"  Verde: {promedio_verde:.2f}")
print(f"  Rojo : {promedio_rojo:.2f}")

canales = {"Azul": promedio_azul, "Verde": promedio_verde, "Rojo": promedio_rojo}
dominante = max(canales, key=canales.get)
print(f"\nCanal dominante en la iluminacion general: {dominante}")