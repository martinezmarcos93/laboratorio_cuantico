"""ml — Pipeline de Machine Learning del Laboratorio Cuántico-Junguiano."""

from .collect_data import (
    generar_dataset_arquetipo,
    generar_dataset_sincronicidad,
    generar_dataset_fidelidad,
)
from .train_regression import (
    entrenar_modelos_arquetipo,
    entrenar_modelo_sincronicidad,
    entrenar_modelo_fidelidad,
    comparar_todos,
)

__all__ = [
    "generar_dataset_arquetipo", "generar_dataset_sincronicidad",
    "generar_dataset_fidelidad",
    "entrenar_modelos_arquetipo", "entrenar_modelo_sincronicidad",
    "entrenar_modelo_fidelidad", "comparar_todos",
]
