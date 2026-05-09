"""
archetypes.py — Módulo de arquetipos junguianos extendidos.

Modela los 5 componentes clásicos de la psique junguiana (Yo, Persona,
Sombra, Ánima/Ánimus, Sí-mismo) como qubits cuánticos con amplitudes
iniciales características de cada arquetipo.

Referencia junguiana:
  - Yo (Ego):      núcleo de la consciencia, identidad estable → P(Ánima) ≈ 0.5 (equilibrio)
  - Persona:       máscara social, adaptación → sesgado hacia polo consciente (|0>)
  - Sombra:        contenidos reprimidos → sesgado hacia polo inconsciente (|1>)
  - Ánima/Ánimus:  contrasexual interno → máxima ambigüedad (superposición perfecta)
  - Sí-mismo:      totalidad psíquica → centro, equilibrado pero con mayor "coherencia"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from experiments import Arquetipo


# ─────────────────────────────────────────────
# Arquetipos individuales
# ─────────────────────────────────────────────

class Persona(Arquetipo):
    """
    La máscara social junguiana.

    Amplitud inicial sesgada hacia |0> (polo adaptado/consciente).
    Alta P(Ánima) ≈ 0.85: la Persona se presenta principalmente
    como el polo socialmente aceptado.
    """
    def __init__(self, seed: int | None = None):
        alpha = np.sqrt(0.85)
        beta  = np.sqrt(0.15)
        super().__init__(alpha, beta, seed=seed)
        self.nombre = "Persona"


class Sombra(Arquetipo):
    """
    Los contenidos reprimidos junguianos.

    Amplitud inicial sesgada hacia |1> (polo inconsciente/reprimido).
    Alta P(Ánimus) ≈ 0.85: la Sombra se manifiesta predominantemente
    como el polo no integrado.
    """
    def __init__(self, seed: int | None = None):
        alpha = np.sqrt(0.15)
        beta  = np.sqrt(0.85)
        super().__init__(alpha, beta, seed=seed)
        self.nombre = "Sombra"


class AnimaAnimus(Arquetipo):
    """
    El contrasexual interno junguiano.

    Estado de máxima superposición (|+> = (|0> + |1>)/√2).
    Entropía máxima = 1 bit. Representa la máxima ambigüedad psíquica:
    el Ánima/Ánimus no está definida hasta el "insight" (colapso).
    """
    def __init__(self, seed: int | None = None):
        super().__init__(1.0, 1.0, seed=seed)   # normaliza a 1/√2 cada una
        self.nombre = "Ánima/Ánimus"


class SiMismo(Arquetipo):
    """
    La totalidad psíquica junguiana (Self).

    Estado perfectamente equilibrado como AnimaAnimus, pero conceptualmente
    representa la integración de todos los opuestos. Aquí se modela idéntico
    a |+> para reflejar que el Sí-mismo trasciende la dualidad.
    """
    def __init__(self, seed: int | None = None):
        super().__init__(1.0, 1.0, seed=seed)
        self.nombre = "SiMismo"


class Yo(Arquetipo):
    """
    El Yo (Ego) junguiano: núcleo estable de la consciencia.

    Levemente sesgado hacia Ánima (consciencia activa) con P(Ánima) ≈ 0.6.
    No es una superposición perfecta: el Yo tiene una identidad definida,
    pero no completamente rígida.
    """
    def __init__(self, seed: int | None = None):
        alpha = np.sqrt(0.60)
        beta  = np.sqrt(0.40)
        super().__init__(alpha, beta, seed=seed)
        self.nombre = "Yo"


# ─────────────────────────────────────────────
# Registro Cuántico: psique completa
# ─────────────────────────────────────────────

class RegistroCuantico:
    """
    Registro de 5 qubits que modela la psique completa junguiana.

    Componentes:
        Yo, Persona, Sombra, Ánima/Ánimus, Sí-mismo

    Cada componente es un Arquetipo independiente con amplitudes
    iniciales características. El registro permite medir la tensión
    Yo↔Sombra, la coherencia global y el estado de individuación.

    La individuación junguiana (integración de la Sombra y el Ánima)
    se mapea a una reducción de entropía diferencial entre componentes.
    """
    COMPONENTES = ["Yo", "Persona", "Sombra", "Ánima/Ánimus", "SiMismo"]

    def __init__(self, seed: int | None = None):
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 9999, size=5)
        self.qubits: list[Arquetipo] = [
            Yo(seed=int(seeds[0])),
            Persona(seed=int(seeds[1])),
            Sombra(seed=int(seeds[2])),
            AnimaAnimus(seed=int(seeds[3])),
            SiMismo(seed=int(seeds[4])),
        ]

    def coherencia_global(self) -> Dict[str, float]:
        """
        Retorna la entropía de Shannon de cada componente (en bits).

        Entropía alta → mayor ambigüedad psíquica (tensión no resuelta).
        Entropía baja → polarización hacia uno de los polos.
        """
        return {
            nombre: qubit.entropia_de_von_neumann()
            for nombre, qubit in zip(self.COMPONENTES, self.qubits)
        }

    def tension_yo_sombra(self) -> float:
        """
        Mide la tensión entre Yo y Sombra como distancia entre sus
        distribuciones de probabilidad.

        Retorna |P(Ánima|Yo) - P(Ánimus|Sombra)| ∈ [0, 1].
        Valor alto → alta tensión no integrada (Sombra muy reprimida).
        Valor bajo → proceso de individuación avanzado.
        """
        yo     = self.qubits[0]
        sombra = self.qubits[2]
        return abs(yo.prob_anima() - (1 - sombra.prob_anima()))

    def medir_todo(self) -> Dict[str, int]:
        """
        Colapsa todos los componentes y retorna el resultado.

        Junguianamente: un "insight" total que colapsa toda la ambigüedad
        psíquica simultáneamente. Los resultados son independientes entre sí
        (no hay entrelazamiento entre componentes en este modelo).
        """
        return {
            nombre: qubit.medir()
            for nombre, qubit in zip(self.COMPONENTES, self.qubits)
        }

    def indice_individuacion(self) -> float:
        """
        Índice de individuación [0, 1]: qué tan integrada está la psique.

        Definido como 1 - tensión_yo_sombra, normalizado por la entropía
        media del registro. Valor 1 = individuación completa (ideal teórico).
        """
        tension   = self.tension_yo_sombra()
        h_media   = np.mean(list(self.coherencia_global().values()))
        # Integración alta cuando tensión baja Y entropía no extrema
        integracion = (1 - tension) * (1 - abs(h_media - 0.5) * 2)
        return float(np.clip(integracion, 0, 1))
