"""
Inteligencia Artificial II - Sesion 9: KNN (K-Vecinos Mas Cercanos)
Solucion del Taller de Laboratorio: Clasificador Universal

Este script:
 1) Transcribe el codigo base del profesor (KNN con 2 caracteristicas y K=3).
 2) Amplia el dataset a 10 clientes con 3 dimensiones (Edad, Salario, Numero de Hijos).
 3) Experimenta con K=1 y K=5 sobre un cliente nuevo.
 4) Deja comentada la reflexion sobre la Maldicion de la Dimensionalidad.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# PARTE 1: Codigo base transcrito (dataset original de la guia)
# ============================================================
print("=" * 60)
print("PARTE 1: Codigo base (dataset de 3 clientes, K=3)")
print("=" * 60)

# 1. Dataset de Entrenamiento: [Caracteristica 1, Caracteristica 2]
X_entrenamiento = np.array([
    [20, 30],  # Punto A
    [40, 50],  # Punto B
    [35, 45]   # Punto C
])

# Etiquetas: 0 = NO COMPRA, 1 = COMPRA
Y_entrenamiento = np.array([0, 1, 1])

# 2. Instanciar el modelo con K = 3
modelo_knn = KNeighborsClassifier(n_neighbors=3)

# 3. "Entrenar" (memorizar los datos)
modelo_knn.fit(X_entrenamiento, Y_entrenamiento)

# 4. Predecir un nuevo punto
nuevo_cliente = np.array([[30, 40]])
prediccion = modelo_knn.predict(nuevo_cliente)

etiquetas_texto = {0: "NO COMPRA", 1: "COMPRA"}
print("Clase predicha:", prediccion[0], "->", etiquetas_texto[prediccion[0]])

# ============================================================
# PARTE 2: Dataset ampliado a 10 clientes y 3 dimensiones
#           (Edad, Salario en miles, Numero de Hijos)
# ============================================================
print("\n" + "=" * 60)
print("PARTE 2: Dataset ampliado (10 clientes, 3 dimensiones)")
print("=" * 60)

# Cada fila: [Edad, Salario (miles), Numero de Hijos]
X_entrenamiento = np.array([
    [20, 30, 0],   # Cliente 1  -> joven, salario bajo, sin hijos
    [40, 50, 2],   # Cliente 2  -> adulto, salario medio-alto, 2 hijos
    [35, 45, 1],   # Cliente 3  -> adulto joven, salario medio, 1 hijo
    [22, 25, 0],   # Cliente 4  -> joven, salario bajo, sin hijos
    [45, 60, 3],   # Cliente 5  -> adulto mayor, salario alto, 3 hijos
    [38, 48, 2],   # Cliente 6  -> adulto, salario medio-alto, 2 hijos
    [19, 20, 0],   # Cliente 7  -> muy joven, salario muy bajo, sin hijos
    [50, 70, 2],   # Cliente 8  -> adulto mayor, salario alto, 2 hijos
    [28, 32, 1],   # Cliente 9  -> joven adulto, salario bajo-medio, 1 hijo
    [42, 55, 3],   # Cliente 10 -> adulto, salario alto, 3 hijos
])

# Etiquetas actualizadas para que coincidan fila a fila con X_entrenamiento
# Regla que seguimos para etiquetar (igual que hizo el profesor con los 3
# clientes originales): salario alto y/o varios hijos -> mas probable que compre
Y_entrenamiento = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 1])

print("Dataset (Edad, Salario en miles, No. Hijos) -> Etiqueta:")
for fila, etiqueta in zip(X_entrenamiento, Y_entrenamiento):
    print(f"  {fila}  ->  {etiquetas_texto[etiqueta]}")

# ============================================================
# PARTE 3: Experimentacion con K=1 y K=5
# ============================================================
print("\n" + "=" * 60)
print("PARTE 3: Experimentacion con distintos valores de K")
print("=" * 60)

# Cliente nuevo a consultar: 33 años, salario 42 (miles), 1 hijo
cliente_nuevo = np.array([[33, 42, 1]])

# --- K = 1 ---
modelo_k1 = KNeighborsClassifier(n_neighbors=1)
modelo_k1.fit(X_entrenamiento, Y_entrenamiento)
prediccion_k1 = modelo_k1.predict(cliente_nuevo)
vecino_dist, vecino_idx = modelo_k1.kneighbors(cliente_nuevo)
print(f"\nCon K=1:")
print(f"  Vecino mas cercano: {X_entrenamiento[vecino_idx[0][0]]} "
      f"(distancia={vecino_dist[0][0]:.2f})")
print(f"  Clase predicha para {cliente_nuevo[0]}: "
      f"{etiquetas_texto[prediccion_k1[0]]}")

# --- K = 5 ---
modelo_k5 = KNeighborsClassifier(n_neighbors=5)
modelo_k5.fit(X_entrenamiento, Y_entrenamiento)
prediccion_k5 = modelo_k5.predict(cliente_nuevo)
vecinos_dist5, vecinos_idx5 = modelo_k5.kneighbors(cliente_nuevo)
print(f"\nCon K=5:")
print("  Los 5 vecinos mas cercanos considerados:")
votos = []
for d, idx in zip(vecinos_dist5[0], vecinos_idx5[0]):
    votos.append(Y_entrenamiento[idx])
    print(f"    Punto {X_entrenamiento[idx]}  distancia={d:.2f}  "
          f"clase={etiquetas_texto[Y_entrenamiento[idx]]}")
print(f"  Votos: {votos.count(1)} COMPRA vs {votos.count(0)} NO COMPRA")
print(f"  Clase predicha para {cliente_nuevo[0]}: "
      f"{etiquetas_texto[prediccion_k5[0]]}")

# ============================================================
# PARTE 4: Pregunta de analisis - La Maldicion de la Dimensionalidad
# ============================================================
print("\n" + "=" * 60)
print("PARTE 4: La Maldicion de la Dimensionalidad (respuesta razonada)")
print("=" * 60)
print("""
Con 3 columnas (Edad, Salario, No. Hijos), la formula de distancia
euclidiana solo suma 3 terminos al cuadrado. Si en vez de 3 columnas
tuvieramos 1,000 (por ejemplo, los 1,000 pixeles de una imagen aplanada),
la formula seria:

    d(P, Q) = raiz( (p1-q1)^2 + (p2-q2)^2 + ... + (p1000-q1000)^2 )

Es la misma formula, solo que ahora sumando 1,000 diferencias al cuadrado
en lugar de 3. Eso NO es gratis: cada dimension adicional aporta su propio
"ruido" a la suma, incluso si esa dimension no es relevante para separar
las clases. En la practica esto provoca que, a medida que crecen las
dimensiones, las distancias entre TODOS los pares de puntos empiecen a
parecerse cada vez mas entre si (el punto mas cercano y el mas lejano
dejan de ser muy distintos en terminos relativos). Si todos los vecinos
quedan "casi igual de lejos", el concepto de "vecino mas cercano" pierde
fuerza discriminativa y KNN empieza a votar casi al azar.
""")

# Verificacion numerica rapida de la intuicion anterior:
# comparamos que tan "parecidas" son las distancias en 3D vs en 1000D
np.random.seed(42)
puntos_3d = np.random.rand(200, 3)
puntos_1000d = np.random.rand(200, 1000)

def razon_dispersión(puntos):
    origen = puntos[0]
    distancias = np.sqrt(np.sum((puntos[1:] - origen) ** 2, axis=1))
    return distancias.min() / distancias.max()

print(f"Relacion (distancia minima / distancia maxima) con 3 dimensiones:    "
      f"{razon_dispersión(puntos_3d):.4f}")
print(f"Relacion (distancia minima / distancia maxima) con 1000 dimensiones: "
      f"{razon_dispersión(puntos_1000d):.4f}")
print("(Entre mas cerca de 1.0 esta esa relacion, menos se distinguen los")
print(" vecinos cercanos de los lejanos: asi se ve la maldicion en numeros).")