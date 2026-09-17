"""
Inteligencia Artificial II - Sesion 11: Redes Neuronales - El Perceptron
Solucion del Taller de Laboratorio: Hackeando los Pesos (La Compuerta OR)

Este script:
 1) Transcribe el codigo base del perceptron (que resuelve la compuerta AND).
 2) Verifica el AND con las 4 combinaciones de entradas.
 3) "Hackea" manualmente los pesos y el sesgo hasta encontrar una combinacion
    que resuelva la compuerta OR.
 4) Verifica el OR con las 4 combinaciones y deja anotados los valores finales.
"""

import numpy as np

# ============================================================
# PARTE 1: Codigo base transcrito (perceptron desde cero)
# ============================================================
print("=" * 60)
print("PARTE 1: Codigo base del Perceptron")
print("=" * 60)


# 1. Definir la Funcion de Activacion (Escalon)
def funcion_escalon(z):
    if z >= 0:
        return 1
    else:
        return 0


# 2. Definir la Estructura de la Neurona
def perceptron(X, W, b):
    # Producto punto (Combinacion lineal)
    # Equivalente a: (X[0]*W[0]) + (X[1]*W[1]) ...
    Z = np.dot(X, W) + b

    # Activacion
    salida = funcion_escalon(Z)
    return salida


# 3. Datos del problema (Compuerta Logica AND)
# El AND solo da 1 si ambas entradas son 1.
entradas = np.array([1, 1])  # Vector X
pesos = np.array([0.5, 0.5])  # Vector W
sesgo = -0.8  # Constante b

# 4. Inferencia (Forward pass)
resultado = perceptron(entradas, pesos, sesgo)
print("El Perceptron disparo el valor:", resultado)

# ============================================================
# PARTE 2: Verificacion del AND con las 4 combinaciones
# ============================================================
print("\n" + "=" * 60)
print("PARTE 2: Verificando la compuerta AND con las 4 combinaciones")
print("=" * 60)

combinaciones = [[1, 1], [1, 0], [0, 1], [0, 0]]
esperado_and = [1, 0, 0, 0]

for entrada, esperado in zip(combinaciones, esperado_and):
    z = np.dot(entrada, pesos) + sesgo
    salida = perceptron(np.array(entrada), pesos, sesgo)
    correcto = "OK" if salida == esperado else "X"
    print(f"  Entrada {entrada}  ->  Z={z:.2f}  ->  Salida={salida}  "
          f"(esperado {esperado})  [{correcto}]")

# ============================================================
# PARTE 3 y 4: "Hackeando" los pesos para resolver la Compuerta OR
# ============================================================
print("\n" + "=" * 60)
print("PARTE 3/4: Buscando pesos y sesgo que resuelvan la Compuerta OR")
print("=" * 60)

# Reglas del OR: da 1 si al menos una entrada es 1; solo da 0 si ambas son 0.
# Razonamiento para encontrar los valores a mano:
#  - Si ambas entradas son 0, Z debe quedar negativo -> Z = b, entonces b < 0.
#  - Si una sola entrada es 1 (ej. [1,0]), ya debe alcanzar para disparar 1
#    -> w1 + b >= 0  (y lo mismo para w2 + b >= 0)
#  - Probando con pesos iguales a los del AND (0.5 y 0.5) pero con un sesgo
#    menos negativo, alcanza con activar la neurona con un solo "1":
pesos_or = np.array([0.5, 0.5])
sesgo_or = -0.4  # <-- este es el "hackeo": subimos el sesgo de -0.8 a -0.4

esperado_or = [1, 1, 1, 0]

print(f"Pesos elegidos: W = {pesos_or.tolist()}   Sesgo elegido: b = {sesgo_or}\n")

for entrada, esperado in zip(combinaciones, esperado_or):
    z = np.dot(entrada, pesos_or) + sesgo_or
    salida = perceptron(np.array(entrada), pesos_or, sesgo_or)
    correcto = "OK" if salida == esperado else "X"
    print(f"  Entrada {entrada}  ->  Z={z:.2f}  ->  Salida={salida}  "
          f"(esperado {esperado})  [{correcto}]")

todo_correcto = all(
    perceptron(np.array(e), pesos_or, sesgo_or) == esp
    for e, esp in zip(combinaciones, esperado_or)
)
print(f"\n¿La compuerta OR quedo resuelta con estos pesos? {todo_correcto}")
print(f"VALORES FINALES ANOTADOS -> W1 = {pesos_or[0]}, W2 = {pesos_or[1]}, b = {sesgo_or}")