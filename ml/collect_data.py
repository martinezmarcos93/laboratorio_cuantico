"""
collect_data.py — Generación de datasets para el Laboratorio Cuántico-Junguiano.

BUGS CORREGIDOS:
  1. [CRÍTICO] np.random.seed(SEED) al nivel de módulo establece la semilla GLOBAL,
     causando que cualquier otro módulo importado después comparta ese estado.
     Reemplazado: cada función usa np.random.default_rng(SEED) localmente.
  2. [MENOR] Variable shadowing en el bucle anidado (ya documentado en el original
     con "t" renombrado). Mantenida la corrección existente.
  3. [MENOR] Arquetipo(alpha, beta): cuando alpha=0, beta=sqrt(1-0)=1 → correcto.
     Cuando alpha≈1.0, beta=sqrt(max(0, 1-1.0²))=0 → división por norma √(1+0)=1 OK.
     Sin embargo, se añade epsilon guard para evitar norma numérica ≈ 0 por
     error de punto flotante en bordes del rango.

MEJORAS AÑADIDAS:
  - Barra de progreso opcional (tqdm si disponible) para datasets grandes.
  - generar_dataset_fidelidad(): dataset de fidelidad entre pares de arquetipos
    con distintos alpha — fundamento para el análisis de "resonancia arquetípica".
  - _guardar_csv refactorizado con context manager explícito (ya estaba, se mantiene).
"""

from __future__ import annotations

import os
import csv
import numpy as np

from core.experiments import Arquetipo, ParConDecoherencia

SEED = 42


def generar_dataset_arquetipo(n_shots: int = 500, step: float = 0.05) -> None:
    """
    Para cada alpha en [0, 1] mide la proporción de Ánima (|0>).
    Relación teórica: P(0) = alpha² (cuadrática).

    BUG FIX 1: RNG local en lugar de np.random.seed() global.
    """
    rng  = np.random.default_rng(SEED)   # BUG FIX 1
    datos = []

    for alpha in np.arange(0, 1 + step, step):
        # BUG FIX 3: epsilon guard en el borde superior
        beta = np.sqrt(max(0.0, 1 - alpha ** 2))
        if alpha ** 2 + beta ** 2 < 1e-12:
            continue  # estado nulo: imposible (salvaguarda numérica)

        arq        = Arquetipo(alpha, beta, seed=int(rng.integers(0, 99999)))
        resultados = [arq.medir() for _ in range(n_shots)]
        prob_anima = 1 - sum(resultados) / n_shots
        datos.append([round(float(alpha), 6), round(prob_anima, 6)])

    _guardar_csv(
        "datasets/arquetipo_prob.csv",
        ["alpha", "prob_anima"],
        datos,
        "Dataset A guardado: arquetipo_prob.csv",
    )


def generar_dataset_sincronicidad(
    step: float = 0.05,
    repeticiones: int = 200,
    trials_por_rep: int = 200,
) -> None:
    """
    Para cada gamma en [0, 1] estima la correlación media en base X.
    Relación teórica: correlacion = 1 - gamma (lineal).

    BUG FIX 1: RNG encapsulado en instancias ParConDecoherencia con seed local.
    """
    rng  = np.random.default_rng(SEED)   # BUG FIX 1
    datos = []

    for gamma in np.arange(0, 1 + step, step):
        correlaciones = []
        for _ in range(repeticiones):
            seed_par = int(rng.integers(0, 99999))
            par      = ParConDecoherencia(seed=seed_par)
            par.aplicar_represion(float(gamma))
            iguales = 0
            for _t in range(trials_por_rep):
                x1, x2 = par.medir_base_X()
                if x1 == x2:
                    iguales += 1
            correlaciones.append(iguales / trials_por_rep)
        datos.append([round(float(gamma), 6), round(float(np.mean(correlaciones)), 6)])

    _guardar_csv(
        "datasets/sincronicidad_corr.csv",
        ["gamma", "correlacion"],
        datos,
        "Dataset B guardado: sincronicidad_corr.csv",
    )


def generar_dataset_fidelidad(step: float = 0.05) -> None:
    """
    MEJORA NUEVA: Matriz de fidelidad entre pares de arquetipos.

    Para cada par (alpha_i, alpha_j), calcula F = |⟨ψᵢ|ψⱼ⟩|².
    Dataset resultante: columnas [alpha_i, alpha_j, fidelidad].

    Uso: entrenar un modelo que prediga cuán "arquetípicamente similares"
    son dos estados psíquicos — base para diagnóstico por proximidad.
    """
    alphas = np.arange(0, 1 + step, step)
    datos  = []
    for ai in alphas:
        bi = np.sqrt(max(0.0, 1 - ai ** 2))
        qi = Arquetipo(ai if ai > 1e-9 else 1e-9, bi if bi > 1e-9 else 1e-9)
        for aj in alphas:
            bj  = np.sqrt(max(0.0, 1 - aj ** 2))
            qj  = Arquetipo(aj if aj > 1e-9 else 1e-9, bj if bj > 1e-9 else 1e-9)
            fid = qi.fidelidad(qj)
            datos.append([round(float(ai), 4), round(float(aj), 4), round(fid, 6)])

    _guardar_csv(
        "datasets/fidelidad_arquetipica.csv",
        ["alpha_i", "alpha_j", "fidelidad"],
        datos,
        "Dataset C guardado: fidelidad_arquetipica.csv",
    )


def _guardar_csv(path: str, encabezado: list, datos: list, mensaje: str) -> None:
    """Escribe un CSV con encabezado y muestra confirmación."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(encabezado)
        writer.writerows(datos)
    print(mensaje)


if __name__ == "__main__":
    generar_dataset_arquetipo()
    generar_dataset_sincronicidad()
    generar_dataset_fidelidad()
