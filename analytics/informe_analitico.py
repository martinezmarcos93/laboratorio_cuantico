"""
informe_analitico.py — Informes clínicos cuántico-junguianos vía Claude API.

Propuesto en INFORME_ANALISIS.md (MEJORA 5 — extensión narrativa).

Toma el estado cuantitativo de un RegistroCuantico (y opcionalmente una
SesionTerapeutica) y genera un informe clínico en lenguaje natural usando
la API de Anthropic.

Requiere:
    pip install anthropic
    Variable de entorno: ANTHROPIC_API_KEY
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.archetypes import RegistroCuantico
    from core.interventions import SesionTerapeutica


# ─────────────────────────────────────────────
# Prompt del sistema
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """
Eres un psicólogo analítico junguiano con sólidos conocimientos de mecánica cuántica computacional.
Interpretas estados cuánticos de arquetipos como representaciones del estado psíquico de un paciente.

GLOSARIO DE CORRESPONDENCIAS:
- P(Ánima) alto → polo consciente/activo dominante
- P(Ánimus) alto → polo inconsciente/pasivo dominante
- Entropía de Shannon alta (→ 1 bit) → ambigüedad activa, superposición de opuestos (favorable para individuación)
- Entropía baja (→ 0 bits) → polarización rígida, arquetipo cristalizado
- Tensión Yo↔Sombra alta → Sombra muy reprimida, escisión psíquica severa
- Índice de individuación alto (→ 1.0) → integración avanzada de opuestos
- Fidelidad alta entre dos arquetipos → resonancia o proyección entre esos componentes

INSTRUCCIONES DE FORMATO:
Estructura tu respuesta en exactamente tres secciones:
1. **Estado global** (2-3 oraciones): síntesis del nivel de individuación
2. **Análisis por componente** (una oración por componente): interpretación clínica de cada qubit
3. **Recomendaciones terapéuticas** (2-3 sugerencias concretas usando las puertas disponibles):
   apertura_consciente (Hadamard), integracion_parcial(θ) (Ry), amplificacion(polo), proyeccion (X)

Sé específico con los valores numéricos pero prioriza la interpretación. Máximo 450 palabras.
Usa lenguaje clínico accesible, no técnico-cuántico en exceso.
""".strip()


# ─────────────────────────────────────────────
# Construcción del prompt de usuario
# ─────────────────────────────────────────────

def _construir_prompt(
    registro: "RegistroCuantico",
    sesion: "SesionTerapeutica | None" = None,
) -> str:
    coherencia = registro.coherencia_global()
    tension    = registro.tension_yo_sombra()
    indice     = registro.indice_individuacion()
    grafo      = registro.grafo_sincronicidad()
    narrativa  = registro.narrativa_estado()
    nombres    = registro.COMPONENTES

    componentes_str = "\n".join(
        f"  • {nom}: P(Ánima)={abs(q.alpha)**2:.4f}, "
        f"P(Ánimus)={abs(q.beta)**2:.4f}, "
        f"H={coherencia[nom]:.4f} bits"
        for nom, q in zip(nombres, registro.qubits)
    )

    # Identificar pares arquetípicos con fidelidad notable
    pares = []
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            f = grafo[i, j]
            if f < 0.25 or f > 0.75:
                tag = "alta resonancia" if f > 0.75 else "alta tensión"
                pares.append(f"  • {nombres[i]} ↔ {nombres[j]}: F={f:.4f} ({tag})")
    fidelidades_str = (
        "Fidelidades inter-arquetípicas notables:\n" + "\n".join(pares)
        if pares else ""
    )

    sesion_str = ""
    if sesion is not None:
        res = sesion.resumen()
        intervenciones = [h["intervencion"] for h in sesion.historial[1:]]
        sesion_str = (
            f"\n--- Sesión terapéutica registrada ---\n"
            f"  Intervenciones aplicadas ({res.get('pasos', 0)}): "
            f"{', '.join(intervenciones)}\n"
            f"  Delta P(Ánima): {res.get('delta_P_anima', 0):+.4f}\n"
            f"  Entropía final: {res.get('entropia_final', 0):.4f} bits\n"
        )

    return (
        f"Diagnóstico cuántico-junguiano del paciente:\n\n"
        f"Resumen narrativo: {narrativa}\n\n"
        f"Estado detallado por componente:\n{componentes_str}\n\n"
        f"Tensión Yo↔Sombra: {tension:.4f}\n"
        f"Índice de individuación: {indice:.4f}\n\n"
        f"{fidelidades_str}"
        f"{sesion_str}\n"
        f"Por favor genera el informe clínico."
    )


# ─────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────

def generar_informe(
    registro: "RegistroCuantico",
    sesion: "SesionTerapeutica | None" = None,
    modelo: str = "claude-sonnet-4-6",
    max_tokens: int = 650,
) -> str:
    """
    Genera un informe clínico narrativo del estado psíquico.

    Args:
        registro:   RegistroCuantico con el estado actual de la psique.
        sesion:     SesionTerapeutica opcional con el historial de intervenciones.
        modelo:     Modelo de Claude a usar (default: claude-sonnet-4-6).
        max_tokens: Longitud máxima del informe.

    Returns:
        Texto del informe clínico generado por el modelo.

    Raises:
        ImportError: si el paquete 'anthropic' no está instalado.
        ValueError:  si ANTHROPIC_API_KEY no está configurada.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "El paquete 'anthropic' no está instalado. "
            "Instalalo con: pip install anthropic"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "La variable de entorno ANTHROPIC_API_KEY no está configurada. "
            "Exportala con: set ANTHROPIC_API_KEY=tu_clave  (Windows CMD)\n"
            "               $env:ANTHROPIC_API_KEY='tu_clave'  (PowerShell)"
        )

    prompt = _construir_prompt(registro, sesion)
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=modelo,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generar_informe_con_cache(
    registro: "RegistroCuantico",
    sesion: "SesionTerapeutica | None" = None,
    modelo: str = "claude-sonnet-4-6",
    max_tokens: int = 650,
) -> str:
    """
    Variante de generar_informe() con prompt caching para reducir latencia
    y costo cuando se generan múltiples informes en la misma sesión de trabajo.

    El system prompt se cachea automáticamente por la API durante 5 minutos.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "El paquete 'anthropic' no está instalado. "
            "Instalalo con: pip install anthropic"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY no configurada.")

    prompt = _construir_prompt(registro, sesion)
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=modelo,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    from core.archetypes    import RegistroCuantico
    from core.interventions import SesionTerapeutica
    from core.experiments   import Arquetipo

    reg    = RegistroCuantico(seed=42)
    arq    = Arquetipo(0.95, float(np.sqrt(1.0 - 0.9025)))
    sesion = SesionTerapeutica(arq)
    sesion.aplicar("apertura_consciente")
    sesion.aplicar("integracion_parcial", theta=np.pi / 4)

    print("Generando informe clínico vía Claude API...\n")
    print("=" * 60)
    informe = generar_informe_con_cache(reg, sesion)
    print(informe)
    print("=" * 60)
