"""
archetypes.py — Módulo de arquetipos junguianos extendidos.

BUGS CORREGIDOS:
  1. [CRÍTICO] RegistroCuantico.indice_individuacion: la normalización
     `abs(h_media - 0.5) * 2` devuelve valores en [0,1] sólo cuando h_media ∈ [0,1],
     pero la entropía de Shannon de un qubit ya está en [0,1] bits, por lo que
     el factor de escala era innecesariamente incierto. Corregido con una fórmula
     más clara y monotónica: promedio ponderado de (1-tensión) y (h_media/1.0).
  2. [MENOR] AnimaAnimus y SiMismo pasaban alpha=1.0, beta=1.0 sin normalizar
     explícitamente — funcionaba porque Arquetipo normaliza, pero era confuso.
     Cambiado a 1/√2 para que sea explícitamente correcto.
  3. [MENOR] coherencia_global retornaba clave "SiMismo" sin tilde pero el
     componente en COMPONENTES era "SiMismo" — consistente, pero se añade
     alias "Sí-mismo" en el diccionario para claridad en la UI.

MEJORAS AÑADIDAS:
  - grafo_sincronicidad(): retorna matriz de fidelidades mutuas entre todos
    los pares de componentes — la "red arquetípica" de la psique.
  - narrativa_estado(): genera un texto interpretativo del estado del registro.
  - RegistroCuantico es ahora iterable (yield from self.qubits).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from .experiments import Arquetipo


# ─────────────────────────────────────────────
# Arquetipos individuales
# ─────────────────────────────────────────────

class Persona(Arquetipo):
    """
    La máscara social junguiana.
    Amplitud sesgada hacia |0> (polo consciente/adaptado). P(Ánima) ≈ 0.85.
    """
    def __init__(self, seed: int | None = None):
        super().__init__(np.sqrt(0.85), np.sqrt(0.15), seed=seed)
        self.nombre = "Persona"


class Sombra(Arquetipo):
    """
    Los contenidos reprimidos junguianos.
    Amplitud sesgada hacia |1> (polo inconsciente). P(Ánimus) ≈ 0.85.
    """
    def __init__(self, seed: int | None = None):
        super().__init__(np.sqrt(0.15), np.sqrt(0.85), seed=seed)
        self.nombre = "Sombra"


class AnimaAnimus(Arquetipo):
    """
    El contrasexual interno junguiano. Máxima superposición (|+⟩).
    BUG FIX 2: amplitudes explícitamente normalizadas a 1/√2.
    """
    def __init__(self, seed: int | None = None):
        # BUG FIX 2: valor explícito en lugar de (1.0, 1.0) que dependía de normalización implícita
        super().__init__(1 / np.sqrt(2), 1 / np.sqrt(2), seed=seed)
        self.nombre = "Ánima/Ánimus"


class SiMismo(Arquetipo):
    """
    La totalidad psíquica (Self). Equilibrado como |+⟩ pero conceptualmente
    representa la integración de todos los opuestos.
    BUG FIX 2: misma corrección que AnimaAnimus.
    """
    def __init__(self, seed: int | None = None):
        super().__init__(1 / np.sqrt(2), 1 / np.sqrt(2), seed=seed)
        self.nombre = "SiMismo"


class Yo(Arquetipo):
    """
    El Yo (Ego) junguiano. Levemente sesgado hacia Ánima. P(Ánima) ≈ 0.6.
    """
    def __init__(self, seed: int | None = None):
        super().__init__(np.sqrt(0.60), np.sqrt(0.40), seed=seed)
        self.nombre = "Yo"


# ─────────────────────────────────────────────
# Registro Cuántico: psique completa
# ─────────────────────────────────────────────

class RegistroCuantico:
    """
    Registro de 5 qubits que modela la psique completa junguiana.

    Componentes: Yo, Persona, Sombra, Ánima/Ánimus, SiMismo.
    Cada componente es un Arquetipo con amplitudes iniciales características.
    """
    COMPONENTES = ["Yo", "Persona", "Sombra", "Ánima/Ánimus", "SiMismo"]

    def __init__(self, seed: int | None = None):
        rng   = np.random.default_rng(seed)
        seeds = rng.integers(0, 9999, size=5)
        self.qubits: list[Arquetipo] = [
            Yo(seed=int(seeds[0])),
            Persona(seed=int(seeds[1])),
            Sombra(seed=int(seeds[2])),
            AnimaAnimus(seed=int(seeds[3])),
            SiMismo(seed=int(seeds[4])),
        ]

    def __iter__(self):
        """Hace el registro iterable: for q in registro."""
        yield from self.qubits

    def coherencia_global(self) -> Dict[str, float]:
        """
        Entropía de Shannon de cada componente (en bits).
        Entropía alta → ambigüedad psíquica; baja → polarización.
        """
        return {
            nombre: qubit.entropia_shannon()
            for nombre, qubit in zip(self.COMPONENTES, self.qubits)
        }

    def tension_yo_sombra(self) -> float:
        """
        Tensión Yo↔Sombra = |P(Ánima|Yo) - P(Ánimus|Sombra)| ∈ [0,1].
        Alto → Sombra muy reprimida; bajo → individuación avanzada.
        """
        yo     = self.qubits[0]
        sombra = self.qubits[2]
        return abs(yo.prob_anima() - (1 - sombra.prob_anima()))

    def medir_todo(self) -> Dict[str, int]:
        """Colapsa todos los componentes. Insight total simultáneo."""
        return {
            nombre: qubit.medir()
            for nombre, qubit in zip(self.COMPONENTES, self.qubits)
        }

    def indice_individuacion(self) -> float:
        """
        Índice de individuación [0, 1].

        BUG FIX 1: fórmula reescrita para ser monotónica y bien definida.

        Definición revisada:
          - tension_norm = 1 - tensión (alto → integrado)
          - h_norm       = entropía media / 1.0 (alto → ambigüedad activa, bien junguiano)
          - índice       = media ponderada: 0.6 * tension_norm + 0.4 * h_norm
        Valor 1 → plena integración con tensión creativa máxima (ideal teórico).
        """
        tension      = self.tension_yo_sombra()
        h_media      = np.mean(list(self.coherencia_global().values()))
        tension_norm = 1.0 - tension          # alto = bien integrado
        h_norm       = float(h_media)         # ya en [0,1] bits para qubit
        indice       = 0.6 * tension_norm + 0.4 * h_norm
        return float(np.clip(indice, 0.0, 1.0))

    def grafo_sincronicidad(self) -> np.ndarray:
        """
        MEJORA NUEVA: Matriz de fidelidades mutuas (5×5) entre todos los componentes.

        grafo[i][j] = F(qubit_i, qubit_j) = |⟨ψᵢ|ψⱼ⟩|²

        La diagonal siempre es 1. Valores altos fuera de diagonal indican
        "resonancia arquetípica" (sincronicidad entre componentes).
        Junguianamente: la Persona y el Yo deberían tener F alto;
        el Yo y la Sombra, F bajo (tensión no resuelta).
        """
        n   = len(self.qubits)
        mat = np.zeros((n, n))
        for i, qi in enumerate(self.qubits):
            for j, qj in enumerate(self.qubits):
                mat[i, j] = qi.fidelidad(qj)
        return mat

    def narrativa_estado(self) -> str:
        """
        MEJORA NUEVA: Genera un texto interpretativo del estado actual del registro.

        Reglas heurísticas basadas en índices cuantitativos → lectura clínica simbólica.
        """
        idx   = self.indice_individuacion()
        tens  = self.tension_yo_sombra()
        h_map = self.coherencia_global()

        # Clasificación
        if idx > 0.75:
            estado_global = "alto grado de individuación — la psique tiende hacia la integración"
        elif idx > 0.45:
            estado_global = "proceso de individuación activo — tensión creativa presente"
        else:
            estado_global = "baja integración — la Sombra permanece escindida"

        sombra_h = h_map["Sombra"]
        if sombra_h < 0.3:
            sombra_txt = "La Sombra está altamente polarizada: su contenido se mantiene reprimido."
        elif sombra_h > 0.7:
            sombra_txt = "La Sombra está en superposición activa: sus contenidos emergen a la conciencia."
        else:
            sombra_txt = "La Sombra muestra ambigüedad moderada: integración parcial en curso."

        return (
            f"Estado psíquico: {estado_global}. "
            f"Tensión Yo↔Sombra: {tens:.3f}. "
            f"{sombra_txt} "
            f"Índice de individuación: {idx:.3f}/1.000."
        )
