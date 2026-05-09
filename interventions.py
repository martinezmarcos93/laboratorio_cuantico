"""
interventions.py — Puertas cuánticas como intervenciones terapéuticas.

Cada función transforma el estado de un Arquetipo aplicando una puerta
cuántica específica, modelando una acción terapéutica junguiana:

  apertura_consciente   → Hadamard (H): lleva al estado a superposición máxima.
                          Jung: apertura al inconsciente, suspensión del juicio.

  integracion_parcial   → Ry(theta): rotación hacia el equilibrio o hacia un polo.
                          Jung: proceso gradual de integración de la Sombra.

  amplificacion         → Proyección reforzada hacia un polo (X o I).
                          Jung: amplificación de un contenido psíquico específico.

  proyeccion            → Pauli-X (NOT): inversión de polo dominante.
                          Jung: proyección total — lo que era consciente se vuelve Sombra.

Todas las funciones retornan un nuevo Arquetipo (inmutabilidad; el estado
anterior queda preservado para el historial de la sesión terapéutica).
"""

import numpy as np
from experiments import Arquetipo


# ─────────────────────────────────────────────
# Puertas individuales
# ─────────────────────────────────────────────

def apertura_consciente(arq: Arquetipo) -> Arquetipo:
    """
    Puerta Hadamard: lleva el estado a superposición máxima.

    H = (1/√2) [[1, 1], [1, -1]]

    Independientemente del estado inicial, el resultado tiene
    P(Ánima) = 0.5 (máxima ambigüedad, máxima entropía).

    Jung: el paciente abre su consciencia al inconsciente sin resistencia;
    suspende toda certeza previa.
    """
    sqrt2_inv = 1 / np.sqrt(2)
    new_alpha = sqrt2_inv * (arq.alpha + arq.beta)
    new_beta  = sqrt2_inv * (arq.alpha - arq.beta)
    obj = object.__new__(Arquetipo)
    obj.alpha, obj.beta = new_alpha, new_beta
    return obj


def integracion_parcial(arq: Arquetipo, theta: float = np.pi / 4) -> Arquetipo:
    """
    Rotación Ry(theta): desplaza el equilibrio entre polos.

    Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]

    theta=0        → sin cambio
    theta=π/2      → rotación de 45°, acerca al equilibrio
    theta=π        → inversión completa de polos (equivale a proyeccion)

    Jung: integración gradual — la sesión terapéutica mueve al paciente
    paso a paso hacia la totalidad, sin forzar un colapso abrupto.
    """
    return arq.aplicar_rotacion(theta)


def amplificacion(arq: Arquetipo, polo: int = 0, theta: float = np.pi / 3) -> Arquetipo:
    """
    Rotación que refuerza un polo específico.

    polo=0 (Ánima): rota hacia |0>, amplifica el polo consciente.
    polo=1 (Ánimus): rota hacia |1>, amplifica el polo inconsciente.

    Jung: la técnica de amplificación junguiana enfatiza un contenido
    psíquico específico para hacerlo más consciente y procesable.
    """
    if polo not in (0, 1):
        raise ValueError(f"polo debe ser 0 (Ánima) o 1 (Ánimus); recibido: {polo}")
    # Rotar hacia polo=0: theta positivo aumenta |0>
    # Rotar hacia polo=1: theta negativo aumenta |1>
    direccion = 1 if polo == 0 else -1
    return arq.aplicar_rotacion(direccion * abs(theta))


def proyeccion(arq: Arquetipo) -> Arquetipo:
    """
    Puerta Pauli-X (NOT): invierte los polos completamente.

    X = [[0, 1], [1, 0]] → α|0> + β|1> se convierte en β|0> + α|1>

    Jung: proyección total — lo que era la Persona (polo dominante)
    se convierte en Sombra, y viceversa. Momento de crisis psíquica
    o de inversión radical de perspectiva.
    """
    obj = object.__new__(Arquetipo)
    obj.alpha, obj.beta = arq.beta, arq.alpha
    return obj


# Alias para la app Streamlit (compatibilidad con selectbox)
def proyeccion_anima(arq: Arquetipo, **kwargs) -> Arquetipo:
    """Amplifica el polo Ánima ignorando parámetros adicionales."""
    return amplificacion(arq, polo=0, theta=kwargs.get("theta", np.pi / 3))

def proyeccion_animus(arq: Arquetipo, **kwargs) -> Arquetipo:
    """Amplifica el polo Ánimus ignorando parámetros adicionales."""
    return amplificacion(arq, polo=1, theta=kwargs.get("theta", np.pi / 3))


# ─────────────────────────────────────────────
# Pipeline terapéutico
# ─────────────────────────────────────────────

class SesionTerapeutica:
    """
    Pipeline de intervenciones secuenciales sobre un Arquetipo.

    Registra el historial de transformaciones para análisis longitudinal.
    El estado se actualiza inmutablemente: cada intervención crea un
    nuevo Arquetipo sin modificar el anterior.

    Analogía junguiana: cada sesión de análisis modifica el estado psíquico
    del paciente; el historial refleja el proceso de individuación.
    """

    INTERVENCIONES = {
        "apertura_consciente":  apertura_consciente,
        "integracion_parcial":  integracion_parcial,
        "amplificacion":        amplificacion,
        "proyeccion":           proyeccion,
        "proyeccion_anima":     proyeccion_anima,
        "proyeccion_animus":    proyeccion_animus,
    }

    def __init__(self, arquetipo_inicial: Arquetipo):
        self.arquetipo = arquetipo_inicial
        self.historial: list[dict] = []
        self._registrar("estado_inicial", arquetipo_inicial)

    def aplicar(self, nombre: str, **kwargs) -> Arquetipo:
        """
        Aplica una intervención por nombre y actualiza el estado.

        Args:
            nombre: Clave en INTERVENCIONES.
            **kwargs: Parámetros adicionales (theta, polo).

        Raises:
            ValueError: Si el nombre de intervención es desconocido.
        """
        if nombre not in self.INTERVENCIONES:
            raise ValueError(
                f"Intervención desconocida: '{nombre}'. "
                f"Disponibles: {list(self.INTERVENCIONES.keys())}"
            )
        fn = self.INTERVENCIONES[nombre]
        nuevo = fn(self.arquetipo, **kwargs)
        self.arquetipo = nuevo
        self._registrar(nombre, nuevo)
        return nuevo

    def _registrar(self, nombre: str, arq: Arquetipo) -> None:
        """Registra el estado en el historial."""
        self.historial.append({
            "intervencion": nombre,
            "alpha":        round(float(arq.alpha), 4),
            "beta":         round(float(arq.beta),  4),
            "P_anima":      round(arq.prob_anima(), 4),
            "P_animus":     round(1 - arq.prob_anima(), 4),
            "entropia":     round(arq.entropia_de_von_neumann(), 4),
        })

    def resumen(self) -> dict:
        """Estadísticas de la sesión: delta total de P(Ánima) y entropía final."""
        if len(self.historial) < 2:
            return {}
        p_ini = self.historial[0]["P_anima"]
        p_fin = self.historial[-1]["P_anima"]
        h_fin = self.historial[-1]["entropia"]
        return {
            "pasos":              len(self.historial) - 1,
            "delta_P_anima":      round(p_fin - p_ini, 4),
            "entropia_final":     h_fin,
            "individuacion":      h_fin > 0.8,   # cerca de superposición máxima
        }
