import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib


def analizar_arquetipo():
    """
    Grafica prob_anima vs alpha con tres curvas:
      - Datos simulados (scatter)
      - Modelo lineal  (regresion_arquetipo_lineal.pkl)
      - Modelo polinomial grado 2 (regresion_arquetipo_poli.pkl)
      - Curva teórica exacta P(0) = alpha²

    Permite comparar visualmente por qué el modelo lineal es inadecuado
    y el polinomial captura la relación real.
    """
    df = pd.read_csv("datasets/arquetipo_prob.csv")
    alpha_range = np.linspace(df["alpha"].min(), df["alpha"].max(), 200).reshape(-1, 1)

    modelo_lineal = joblib.load("modelos/regresion_arquetipo_lineal.pkl")
    modelo_poli   = joblib.load("modelos/regresion_arquetipo_poli.pkl")

    plt.figure(figsize=(8, 5))
    plt.scatter(df["alpha"], df["prob_anima"], alpha=0.6, label="Datos simulados", zorder=3)
    plt.plot(alpha_range, modelo_lineal.predict(alpha_range), "r--", label="Modelo lineal")
    plt.plot(alpha_range, modelo_poli.predict(alpha_range),   "g-",  label="Modelo polinomial (grado 2)")
    plt.plot(alpha_range, alpha_range**2,                     "k:",  label="Teórico: P(0) = α²", linewidth=1.5)

    plt.xlabel("alpha (amplitud Ánima)")
    plt.ylabel("Probabilidad observada de Ánima")
    plt.title("Superposición arquetípica")
    plt.legend()
    plt.tight_layout()
    plt.show()


def analizar_sincronicidad():
    """
    Grafica correlacion vs gamma con:
      - Datos simulados (scatter)
      - Predicción del modelo lineal (regresion_sincronicidad.pkl)
      - Curva teórica exacta: correlación = 1 - gamma

    La relación teórica exacta del canal de desfase implementado en
    ParConDecoherencia.aplicar_represion() es correlacion = 1 - gamma
    (perfectamente lineal), verificada numéricamente.
    """
    df = pd.read_csv("datasets/sincronicidad_corr.csv")
    gamma_range = np.linspace(df["gamma"].min(), df["gamma"].max(), 200).reshape(-1, 1)

    modelo = joblib.load("modelos/regresion_sincronicidad.pkl")

    plt.figure(figsize=(8, 5))
    plt.scatter(df["gamma"], df["correlacion"], alpha=0.6, label="Datos simulados", zorder=3)
    plt.plot(gamma_range, modelo.predict(gamma_range), "r--", label="Modelo lineal")
    # BUG FIX: la curva teórica era mencionada en el docstring pero nunca se graficaba.
    # La fórmula correcta (verificada numéricamente) es correlacion = 1 - gamma.
    plt.plot(gamma_range, 1 - gamma_range,             "k:",  label="Teórico: 1 − γ", linewidth=1.5)

    plt.xlabel("Gamma (nivel de represión)")
    plt.ylabel("Correlación en base X")
    plt.title("Sincronicidad bajo decoherencia")
    plt.legend()
    plt.tight_layout()
    plt.show()


def analizar_ambos():
    """Ejecuta los dos análisis en secuencia."""
    analizar_arquetipo()
    analizar_sincronicidad()


if __name__ == "__main__":
    analizar_ambos()
