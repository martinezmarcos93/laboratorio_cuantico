import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

SEED       = 42
TEST_SIZE  = 0.2


def entrenar_modelos_arquetipo(dataset_path='datasets/arquetipo_prob.csv'):
    """
    Entrena y compara un modelo lineal y uno polinomial (grado 2) sobre el
    dataset del arquetipo.

    La relación teórica es P(0) = alpha² (cuadrática), por lo que se espera
    que el modelo polinomial supere claramente al lineal en R².

    Guarda ambos modelos en modelos/ y retorna (modelo_lineal, modelo_poli).
    """
    df = pd.read_csv(dataset_path)
    X  = df[['alpha']].values
    y  = df['prob_anima'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # Modelo lineal
    lin = LinearRegression()
    lin.fit(X_train, y_train)

    # Modelo polinomial grado 2
    poli = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
    poli.fit(X_train, y_train)

    _reportar('Arquetipo — Lineal',      lin,  X_train, X_test, y_train, y_test)
    _reportar('Arquetipo — Polinomial²', poli, X_train, X_test, y_train, y_test)

    os.makedirs('modelos', exist_ok=True)
    joblib.dump(lin,  'modelos/regresion_arquetipo_lineal.pkl')
    joblib.dump(poli, 'modelos/regresion_arquetipo_poli.pkl')
    print("Modelos del arquetipo guardados.\n")

    return lin, poli


def entrenar_modelo_sincronicidad(dataset_path='datasets/sincronicidad_corr.csv'):
    """
    Entrena un modelo lineal sobre el dataset de sincronicidad.

    La relación teórica es correlacion = 1 - gamma (exactamente lineal),
    por lo que se espera R² ≈ 1.0 tanto en train como en test.

    Guarda el modelo en modelos/ y lo retorna.
    """
    df = pd.read_csv(dataset_path)
    X  = df[['gamma']].values
    y  = df['correlacion'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    lin = LinearRegression()
    lin.fit(X_train, y_train)

    _reportar('Sincronicidad — Lineal', lin, X_train, X_test, y_train, y_test)

    os.makedirs('modelos', exist_ok=True)
    joblib.dump(lin, 'modelos/regresion_sincronicidad.pkl')
    print("Modelo de sincronicidad guardado.\n")

    return lin


def _reportar(nombre, modelo, X_train, X_test, y_train, y_test):
    """Imprime métricas de train y test para un modelo dado."""
    r2_train  = r2_score(y_train, modelo.predict(X_train))
    r2_test   = r2_score(y_test,  modelo.predict(X_test))
    mse_train = mean_squared_error(y_train, modelo.predict(X_train))
    mse_test  = mean_squared_error(y_test,  modelo.predict(X_test))

    print(f"--- {nombre} ---")
    print(f"  Train → R²: {r2_train:.4f}  MSE: {mse_train:.6f}")
    print(f"  Test  → R²: {r2_test:.4f}  MSE: {mse_test:.6f}")


if __name__ == '__main__':
    entrenar_modelos_arquetipo()
    entrenar_modelo_sincronicidad()
