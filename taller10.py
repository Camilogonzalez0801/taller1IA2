"""
Inteligencia Artificial II - Sesion 10: SVM (Support Vector Machine)
Solucion del Taller de Laboratorio: Fronteras No Lineales

Este script:
 1) Transcribe el codigo base del profesor (SVM lineal, dataset de 6 puntos).
 2) Agrega un punto "trampa" [5,5] etiquetado como Clase A para forzar la frontera lineal.
 3) Reentrena con kernel='linear' y observa que tan forzada queda la recta.
 4) Cambia a kernel='rbf' y reentrena/predice.
 5) Deja la reflexion sobre cuando un kernel lineal no alcanza.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# ============================================================
# PARTE 1: Codigo base transcrito (dataset original, kernel lineal)
# ============================================================
print("=" * 60)
print("PARTE 1: Codigo base (SVM lineal, 6 puntos)")
print("=" * 60)

# 1. Crear el dataset (X = Coordenadas, Y = Etiquetas binarias 0 o 1)
X = np.array([[2, 2], [3, 3], [4, 2], [6, 6], [7, 8], [8, 7]])
Y = np.array([0, 0, 0, 1, 1, 1])

# 2. Inicializar SVM con Kernel Lineal
modelo_svm = SVC(kernel="linear")

# 3. Entrenar el modelo (aprender la ecuacion del hiperplano)
modelo_svm.fit(X, Y)

# 4. Extraer los Vectores de Soporte descubiertos por la IA
vectores = modelo_svm.support_vectors_
print("Los Vectores de Soporte son:\n", vectores)

# 5. Prediccion
nuevo_punto = np.array([[5, 4]])
pred = modelo_svm.predict(nuevo_punto)
print("El punto [5,4] pertenece a la clase:", pred[0])

# Verificacion: ¿coinciden con los que se marcarian a mano en el taller analitico?
w = modelo_svm.coef_[0]
b = modelo_svm.intercept_[0]
print(f"\nEcuacion del hiperplano: {w[0]:.4f}*x + {w[1]:.4f}*y + {b:.4f} = 0")
print(f"(equivale aproximadamente a la recta  x + y = 9)")
print(f"Ancho del margen (la 'calle'): {2 / np.linalg.norm(w):.4f}")

# Grafica del taller analitico: margen, hiperplano y vectores de soporte
fig, ax = plt.subplots(figsize=(6, 6))
claseA, claseB = X[Y == 0], X[Y == 1]
ax.scatter(claseA[:, 0], claseA[:, 1], marker="o", s=140, c="#1f77b4", label="Clase A (circulos)", zorder=3)
ax.scatter(claseB[:, 0], claseB[:, 1], marker="x", s=160, c="#d62728", linewidths=3, label="Clase B (equis)", zorder=3)
xx = np.linspace(0, 10, 200)
yy = -(w[0] * xx + b) / w[1]
margen = 1 / np.linalg.norm(w)
factor = np.sqrt(1 + (w[0] / w[1]) ** 2)
ax.plot(xx, yy, "k-", linewidth=2, label="Hiperplano (x + y = 9)")
ax.plot(xx, yy + margen * factor, "k--", linewidth=1, alpha=0.6)
ax.plot(xx, yy - margen * factor, "k--", linewidth=1, alpha=0.6)
sv = modelo_svm.support_vectors_
ax.scatter(sv[:, 0], sv[:, 1], s=400, facecolors="none", edgecolors="red", linewidths=2.2, label="Vectores de Soporte", zorder=4)
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_xlabel("X"); ax.set_ylabel("Y")
ax.set_title("Taller Analitico: Margen optimo y Vectores de Soporte")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("1_margen_analitico.png", dpi=150)
print("Grafica guardada: 1_margen_analitico.png")

# ============================================================
# PARTE 2 y 3: Punto "trampa" [5,5] como Clase A, reentrenar lineal
# ============================================================
print("\n" + "=" * 60)
print("PARTE 2/3: Agregamos el punto trampa [5,5] (Clase A) y reentrenamos lineal")
print("=" * 60)

X_trampa = np.array([[2, 2], [3, 3], [4, 2], [6, 6], [7, 8], [8, 7], [5, 5]])
Y_trampa = np.array([0, 0, 0, 1, 1, 1, 0])

modelo_lineal_trampa = SVC(kernel="linear")
modelo_lineal_trampa.fit(X_trampa, Y_trampa)

w2 = modelo_lineal_trampa.coef_[0]
b2 = modelo_lineal_trampa.intercept_[0]
print("Nuevos vectores de soporte:\n", modelo_lineal_trampa.support_vectors_)
print(f"Nueva ecuacion: {w2[0]:.4f}*x + {w2[1]:.4f}*y + {b2:.4f} = 0  "
      f"(equivale aprox. a x + y = 11)")
print(f"Nuevo ancho del margen (calle): {2 / np.linalg.norm(w2):.4f}  "
      f"<-- mucho mas angosto que el 4.24 original")

pred_dataset = modelo_lineal_trampa.predict(X_trampa)
print("Precision sobre el propio dataset:", modelo_lineal_trampa.score(X_trampa, Y_trampa))
print("Predicciones :", pred_dataset)
print("Reales       :", Y_trampa)
pred_trampa_lineal = modelo_lineal_trampa.predict(nuevo_punto)
print("Prediccion lineal (con trampa) para [5,4]:", pred_trampa_lineal[0])

# ============================================================
# PARTE 4: Cambiar el kernel a 'rbf'
# ============================================================
print("\n" + "=" * 60)
print("PARTE 4: Mismo dataset (con trampa), pero kernel='rbf'")
print("=" * 60)

modelo_rbf = SVC(kernel="rbf")
modelo_rbf.fit(X_trampa, Y_trampa)

pred_dataset_rbf = modelo_rbf.predict(X_trampa)
print("Precision sobre el propio dataset (rbf):", modelo_rbf.score(X_trampa, Y_trampa))
print("Predicciones (rbf):", pred_dataset_rbf)
print("Numero de vectores de soporte (rbf):", modelo_rbf.support_vectors_.shape[0])

pred_rbf = modelo_rbf.predict(nuevo_punto)
print("Prediccion rbf para [5,4]:", pred_rbf[0])

# Grafica comparativa: frontera lineal forzada vs frontera curva (rbf)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
xx_grid, yy_grid = np.meshgrid(np.linspace(0, 10, 300), np.linspace(0, 10, 300))
grid_puntos = np.c_[xx_grid.ravel(), yy_grid.ravel()]

for ax, kernel, titulo in zip(
    axes, ["linear", "rbf"],
    ["Kernel lineal (forzado por el punto trampa)", "Kernel 'rbf' (frontera curva)"],
):
    m = SVC(kernel=kernel)
    m.fit(X_trampa, Y_trampa)
    Z = m.predict(grid_puntos).reshape(xx_grid.shape)
    ax.contourf(xx_grid, yy_grid, Z, alpha=0.25, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
    cA, cB = X_trampa[Y_trampa == 0], X_trampa[Y_trampa == 1]
    ax.scatter(cA[:, 0], cA[:, 1], marker="o", s=140, c="#1f77b4", label="Clase A", zorder=3)
    ax.scatter(cB[:, 0], cB[:, 1], marker="x", s=160, c="#d62728", linewidths=3, label="Clase B", zorder=3)
    ax.scatter(5, 5, marker="*", s=300, c="gold", edgecolors="black", linewidths=1, label="Punto trampa [5,5]", zorder=5)
    sv2 = m.support_vectors_
    ax.scatter(sv2[:, 0], sv2[:, 1], s=350, facecolors="none", edgecolors="black", linewidths=1.8, label="Vectores de Soporte", zorder=4)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("2_comparacion_lineal_rbf.png", dpi=150)
print("Grafica guardada: 2_comparacion_lineal_rbf.png")

# ============================================================
# PARTE 5: Reflexion (queda como comentario/print, no es codigo ejecutable)
# ============================================================
print("\n" + "=" * 60)
print("PARTE 5: Reflexion - ¿Cuando un kernel lineal fallaria completamente?")
print("=" * 60)
print("""
Un kernel lineal solo puede trazar una recta (o un hiperplano plano en mas
dimensiones). Eso funciona bien si las clases se pueden separar con una
sola raya, pero falla por completo cuando una clase esta "rodeada" por la
otra, formando un patron circular/anular en vez de dos grupos separados.

Ejemplo en medicina: en un diagnostico basado en dos biomarcadores (por
ejemplo, dos hormonas en sangre), a veces los pacientes "sanos" caen en un
rango medio central mientras que los pacientes "enfermos" caen tanto en
valores muy bajos como muy altos (ambos extremos son anomalos). Graficado
en 2D, los sanos forman un circulo en el centro y los enfermos rodean ese
circulo por todos lados. Ninguna linea recta puede separar "el circulo de
adentro" del "anillo de afuera": la unica forma de aislarlos es con una
frontera curva, que es exactamente lo que logra el kernel RBF al proyectar
los datos a una dimension donde si son linealmente separables.

Otro ejemplo real es el reconocimiento facial: las variaciones de luz,
angulo y expresion hacen que los rostros de una misma persona no formen
una nube compacta y separable con una sola recta de las demas personas;
las fronteras entre identidades terminan siendo curvas e irregulares, por
lo que en la practica casi siempre se usa RBF (o kernels aun mas complejos)
en vez de un kernel lineal.
""")