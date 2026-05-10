"""
config.py — Rutas centralizadas del Laboratorio Cuántico-Junguiano.

Importar desde cualquier módulo para obtener rutas absolutas correctas
independientemente del directorio de trabajo actual.
"""

from pathlib import Path

ROOT_DIR     = Path(__file__).parent
DATASETS_DIR = ROOT_DIR / "datasets"
MODELOS_DIR  = ROOT_DIR / "modelos"
DOCS_DIR     = ROOT_DIR / "docs"

# Datasets
DATASET_ARQUETIPO    = DATASETS_DIR / "arquetipo_prob.csv"
DATASET_SINCRONICIDAD = DATASETS_DIR / "sincronicidad_corr.csv"
DATASET_FIDELIDAD    = DATASETS_DIR / "fidelidad_arquetipica.csv"

# Modelos
MODELO_LINEAL        = MODELOS_DIR / "regresion_arquetipo_lineal.pkl"
MODELO_POLI          = MODELOS_DIR / "regresion_arquetipo_poli.pkl"
MODELO_SINCRONICIDAD = MODELOS_DIR / "regresion_sincronicidad.pkl"
MODELO_FIDELIDAD_MLP = MODELOS_DIR / "regresion_fidelidad_mlp.pkl"
