import numpy as np

# Semilla fija para que los numeros aleatorios sean reproducibles
# (si se quita esta linea, la matriz aleatoria cambiara cada vez que se ejecute el script)
np.random.seed(42)


# ============================================================
# TALLER DE LABORATORIO 1: TRANSFORMACIONES AFINES
# ============================================================
print("=" * 60)
print("TALLER DE LABORATORIO 1: TRANSFORMACIONES AFINES")
print("=" * 60)

# 1. Matriz de prueba 5x5 que simula una imagen sobreexpuesta
#    (valores altos, cercanos a 255 = blanco)
matriz_original = np.random.randint(200, 255, (5, 5))
print("\nMatriz ORIGINAL (imagen sobreexpuesta):")
print(matriz_original)

# 2. Parametros de la transformacion afin A_nueva = alpha*A + beta
alpha = 0.5    # Reduccion de contraste del 50%  -> multiplica el rango a la mitad
beta = -50.0   # Disminucion de brillo en 50 unidades -> resta 50 a cada pixel

# 3. Aplicamos la transformacion lineal (todavia en punto flotante, sin recortar)
matriz_transformada = (alpha * matriz_original) + beta
print("\nMatriz TRANSFORMADA (antes de saturar, valores float):")
print(matriz_transformada)

# 4. Saturacion (clipping) para volver a un rango valido de imagen [0, 255]
#    y conversion a enteros de 8 bits sin signo (formato real de pixel)
matriz_final = np.clip(matriz_transformada, 0, 255).astype(np.uint8)

print("\nMatriz FINAL (saturada con np.clip y convertida a np.uint8):")
print(matriz_final)



# TALLER DE LABORATORIO FINAL: PROGRAMANDO UN KERNEL (CONVOLUCION)
print("\n" + "=" * 60)
print("TALLER DE LABORATORIO FINAL: PROGRAMANDO UN KERNEL")
print("=" * 60)

# 1. Seccion de imagen (I) y Kernel de realce de bordes (K), tal como
#    aparecen en el diagrama de la guia
I = np.array([
    [100, 100, 100],
    [100, 200, 100],
    [100, 100, 100]
])

K = np.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
])

print("\nSeccion de Imagen (I):")
print(I)
print("\nKernel (K) de realce:")
print(K)

# 2. Producto Hadamard (elemento a elemento) entre I y K
producto_hadamard = I * K
print("\nProducto Hadamard (I * K), elemento a elemento:")
print(producto_hadamard)

# 3. Suma de TODOS los elementos del resultado -> esto es la convolucion
#    en ese unico punto (el pixel central de la seccion de imagen)
pixel_central_resultante = np.sum(producto_hadamard)
print("\nValor del pixel central resultante (np.sum del producto Hadamard):")
print(pixel_central_resultante)