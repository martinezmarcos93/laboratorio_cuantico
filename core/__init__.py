"""core — Núcleo cuántico del Laboratorio Cuántico-Junguiano."""

from .experiments import Arquetipo, ParConDecoherencia
from .archetypes import (
    Persona, Sombra, AnimaAnimus, SiMismo, Yo, RegistroCuantico,
)
from .interventions import (
    apertura_consciente, apertura_consciente_inversa,
    integracion_parcial, amplificacion, proyeccion,
    proyeccion_anima, proyeccion_animus,
    SesionTerapeutica,
)

__all__ = [
    "Arquetipo", "ParConDecoherencia",
    "Persona", "Sombra", "AnimaAnimus", "SiMismo", "Yo", "RegistroCuantico",
    "apertura_consciente", "apertura_consciente_inversa",
    "integracion_parcial", "amplificacion", "proyeccion",
    "proyeccion_anima", "proyeccion_animus",
    "SesionTerapeutica",
]
