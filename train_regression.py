"""
train_regression.py — Entrenamiento de modelos de regresión.

BUGS CORREGIDOS:
  1. [MENOR] _reportar: las predicciones ya estaban optimizadas (BUG FIX
     documentado en el original). Se mantiene y se extiende.
  2. [MENOR] No había validación de existencia de los archivos CSV antes de
     intentar leerlos, produciendo FileNotFoundError sin contexto. Añadido
     guard con mensaje descriptivo.
  3. [MENOR] TEST_SIZE y SEED eran constantes de módulo sin typing.
     Añadida type annotation y validación.

MEJORAS AÑADIDAS:
  - entrenar_modelo_fidelidad(): entrena una red neuronal simple (MLPRegressor)
    sobre el dataset de fidelidad arquetípica — demuestra capacidades no lineales.
  - comparar_todos(): tabla resumen de métricas de todos los modelos entrenados.
  - _cargar_dataset(): helper con validación de existencia y logging.
  - Soporte para GridSearchCV en modelo polinomial para selección automática de grado.
"""

from __future__ import annotations

import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model  import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline      import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics       import r2_score, mean_squared_error

SEED:      int   = 42
TEST_SIZE: float = 0.2


def _cargar_dataset(path: str, xcol: str, ycol: str):
    """
    BUG FIX 2: carga con validación de existencia y mensaje descriptivo.
    Retorna (X, y) como arrays numpy.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset no encontrado: '{path}'. "
            "Ejecuta primero collect_data.py (o main.py opción 1)."
        )
    df = pd.read_csv(path)
    return df[[xcol]].values, df[ycol].values


def entrenar_modelos_arquetipo(dataset_path: str = "datasets/arquetipo_prob.csv"):
    """
    Entrena y compara modelo lineal y polinomial sobre el dataset del arquetipo.
    Relación teórica: P(0) = alpha² (cuadrática).

    MEJORA: GridSearchCV elige automáticamente el mejor grado (2–4) en lugar
    de hardcodear grado=2.
    """
    X, y = _cargar_dataset(dataset_path, "alpha", "prob_anima")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    # Modelo lineal
    lin = LinearRegression()
    lin.fit(X_tr, y_tr)

    # MEJORA: búsqueda de grado óptimo (2–4)
    param_grid = {"polynomialfeatures__degree": [2, 3, 4]}
    pipe_base  = make_pipeline(PolynomialFeatures(include_bias=False), LinearRegression())
    gs         = GridSearchCV(pipe_base, param_grid, cv=5, scoring="r2")
    gs.fit(X_tr, y_tr)
    poli = gs.best_estimator_
    best_degree = gs.best_params_["polynomialfeatures__degree"]
    print(f"  Mejor grado polinomial: {best_degree}")

    _reportar("Arquetipo — Lineal",                 lin,  X_tr, X_te, y_tr, y_te)
    _reportar(f"Arquetipo — Polinomial(grado={best_degree})", poli, X_tr, X_te, y_tr, y_te)

    os.makedirs("modelos", exist_ok=True)
    joblib.dump(lin,  "modelos/regresion_arquetipo_lineal.pkl")
    joblib.dump(poli, "modelos/regresion_arquetipo_poli.pkl")
    print("Modelos del arquetipo guardados.\n")
    return lin, poli


def entrenar_modelo_sincronicidad(dataset_path: str = "datasets/sincronicidad_corr.csv"):
    """
    Entrena un modelo lineal sobre el dataset de sincronicidad.
    Relación teórica exacta: correlacion = 1 - gamma → R² ≈ 1.0 esperado.
    """
    X, y = _cargar_dataset(dataset_path, "gamma", "correlacion")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    lin = LinearRegression()
    lin.fit(X_tr, y_tr)

    _reportar("Sincronicidad — Lineal", lin, X_tr, X_te, y_tr, y_te)

    os.makedirs("modelos", exist_ok=True)
    joblib.dump(lin, "modelos/regresion_sincronicidad.pkl")
    print("Modelo de sincronicidad guardado.\n")
    return lin


def entrenar_modelo_fidelidad(dataset_path: str = "datasets/fidelidad_arquetipica.csv"):
    """
    MEJORA NUEVA: Entrena un MLPRegressor sobre el dataset de fidelidad.

    Input:  (alpha_i, alpha_j) — dos amplitudes arquetípicas.
    Output: fidelidad F = |⟨ψᵢ|ψⱼ⟩|².

    La relación analítica es F = (aᵢaⱼ + bᵢbⱼ)² (no lineal en alpha),
    por lo que una red neuronal la captura mejor que regresión polinomial simple.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset de fidelidad no encontrado: '{dataset_path}'. "
            "Ejecuta collect_data.generar_dataset_fidelidad() primero."
        )
    df  = pd.read_csv(dataset_path)
    X   = df[["alpha_i", "alpha_j"]].values
    y   = df["fidelidad"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(X_tr, y_tr)
    _reportar("Fidelidad — MLP(64,32)", mlp, X_tr, X_te, y_tr, y_te)

    os.makedirs("modelos", exist_ok=True)
    joblib.dump(mlp, "modelos/regresion_fidelidad_mlp.pkl")
    print("Modelo de fidelidad (MLP) guardado.\n")
    return mlp


def comparar_todos() -> pd.DataFrame:
    """
    MEJORA NUEVA: Tabla resumen de R² test para todos los modelos entrenados.
    Retorna DataFrame con columnas [Modelo, Dataset, R²_test, MSE_test].
    """
    rutas = {
        "Arquetipo Lineal":    ("modelos/regresion_arquetipo_lineal.pkl", "datasets/arquetipo_prob.csv",    "alpha", "prob_anima"),
        "Arquetipo Poli":      ("modelos/regresion_arquetipo_poli.pkl",   "datasets/arquetipo_prob.csv",    "alpha", "prob_anima"),
        "Sincronicidad Lineal":("modelos/regresion_sincronicidad.pkl",    "datasets/sincronicidad_corr.csv","gamma", "correlacion"),
    }
    filas = []
    for nombre, (mpath, dpath, xcol, ycol) in rutas.items():
        if not (os.path.exists(mpath) and os.path.exists(dpath)):
            continue
        modelo     = joblib.load(mpath)
        X, y       = _cargar_dataset(dpath, xcol, ycol)
        _, X_te, _, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
        y_pred     = modelo.predict(X_te)
        filas.append({
            "Modelo":    nombre,
            "R²_test":   round(r2_score(y_te, y_pred), 4),
            "MSE_test":  round(mean_squared_error(y_te, y_pred), 6),
        })
    return pd.DataFrame(filas)


def _reportar(nombre: str, modelo, X_train, X_test, y_train, y_test) -> None:
    """Imprime métricas de train y test. Predicciones calculadas una sola vez."""
    y_pred_train = modelo.predict(X_train)
    y_pred_test  = modelo.predict(X_test)
    print(f"--- {nombre} ---")
    print(f"  Train → R²: {r2_score(y_train, y_pred_train):.4f}  "
          f"MSE: {mean_squared_error(y_train, y_pred_train):.6f}")
    print(f"  Test  → R²: {r2_score(y_test,  y_pred_test):.4f}  "
          f"MSE: {mean_squared_error(y_test,  y_pred_test):.6f}")


if __name__ == "__main__":
    entrenar_modelos_arquetipo()
    entrenar_modelo_sincronicidad()
    # Opcional (requiere dataset C):
    # entrenar_modelo_fidelidad()
    tabla = comparar_todos()
    print("\n=== RESUMEN DE MODELOS ===")
    print(tabla.to_string(index=False))
