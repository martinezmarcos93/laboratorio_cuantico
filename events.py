"""
events.py — Event Sourcing para el proceso de individuación.

Propuesto en INFORME_ANALISIS.md (MEJORA 6).

DiarioIndividuacion persiste sesiones terapéuticas en formato JSONL
(una línea = una sesión completa), permitiendo:
  - Replay completo de la trayectoria psíquica de un paciente.
  - Análisis longitudinal: cómo evoluciona la entropía sesión a sesión.
  - Comparación entre pacientes o entre configuraciones.

La elección de JSONL (JSON Lines) permite streams sin límite de tamaño
y procesamiento línea a línea sin cargar todo en memoria.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from interventions import SesionTerapeutica


class DiarioIndividuacion:
    """
    Event Store para el proceso de individuación.

    Cada entrada registra una sesión completa con timestamp, ID de paciente
    y notas del terapeuta. El archivo JSONL permite append eficiente y
    replay completo en cualquier momento.
    """

    def __init__(self, path: str = "diario_individuacion.jsonl"):
        self.path = pathlib.Path(path)

    # ─── escritura ─────────────────────────────────────────────

    def registrar(
        self,
        sesion: "SesionTerapeutica",
        paciente: str = "anónimo",
        notas: str = "",
    ) -> None:
        """
        Persiste una SesionTerapeutica ya ejecutada en el diario.

        Args:
            sesion:   sesión completada con al menos un paso.
            paciente: identificador del paciente (string libre).
            notas:    observaciones clínicas opcionales.
        """
        datos = json.loads(sesion.exportar_json())
        entrada = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "paciente":  paciente,
            "notas":     notas,
            **datos,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")

    # ─── lectura ────────────────────────────────────────────────

    def replay(self) -> list[dict]:
        """Devuelve todas las sesiones en orden cronológico de registro."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(linea) for linea in f if linea.strip()]

    def trayectoria(self, paciente: str) -> list[dict]:
        """Todas las sesiones de un paciente, en orden cronológico."""
        return [e for e in self.replay() if e.get("paciente") == paciente]

    def pacientes(self) -> list[str]:
        """Lista de pacientes únicos registrados en el diario."""
        return sorted({e.get("paciente", "anónimo") for e in self.replay()})

    def ultima_sesion(self, paciente: str) -> dict | None:
        """Devuelve la sesión más reciente de un paciente."""
        tray = self.trayectoria(paciente)
        return tray[-1] if tray else None

    # ─── análisis longitudinal ──────────────────────────────────

    def estadisticas_longitudinales(self, paciente: str) -> dict:
        """
        Métricas de evolución acumulada a lo largo de las sesiones del paciente.

        Retorna:
            sesiones          — número total de sesiones
            entropia_media    — media de la entropía final por sesión
            entropia_tendencia— pendiente de la entropía (positivo = mejora)
            delta_p_acumulado — suma de deltas de P(Ánima) a lo largo del tiempo
            tendencia         — etiqueta cualitativa del proceso
        """
        tray = self.trayectoria(paciente)
        if not tray:
            return {"sesiones": 0}

        entropias_finales: list[float] = []
        deltas_p: list[float] = []

        for entrada in tray:
            hist = entrada.get("historial", [])
            if not hist:
                continue
            entropias_finales.append(float(hist[-1].get("entropia", 0.0)))
            if len(hist) >= 2:
                delta = float(hist[-1].get("P_anima", 0)) - float(hist[0].get("P_anima", 0))
                deltas_p.append(delta)

        if not entropias_finales:
            return {"sesiones": len(tray)}

        h_media   = float(np.mean(entropias_finales))
        delta_tot = float(np.sum(deltas_p)) if deltas_p else 0.0

        # Pendiente lineal de la entropía si hay >= 3 sesiones
        if len(entropias_finales) >= 3:
            xs = np.arange(len(entropias_finales), dtype=float)
            pendiente = float(np.polyfit(xs, entropias_finales, 1)[0])
        else:
            pendiente = 0.0

        tendencia = (
            "individuación activa"   if h_media > 0.7 else
            "proceso en marcha"      if h_media > 0.4 else
            "polarización persistente"
        )

        return {
            "sesiones":             len(tray),
            "entropia_media":       round(h_media, 4),
            "entropia_tendencia":   round(pendiente, 4),
            "delta_p_acumulado":    round(delta_tot, 4),
            "tendencia":            tendencia,
        }

    def serie_temporal_entropia(self, paciente: str) -> list[float]:
        """
        Devuelve la entropía final de cada sesión como serie temporal.
        Útil para graficar la trayectoria de individuación.
        """
        tray = self.trayectoria(paciente)
        entropias = []
        for entrada in tray:
            hist = entrada.get("historial", [])
            if hist:
                entropias.append(float(hist[-1].get("entropia", 0.0)))
        return entropias

    # ─── visualización ──────────────────────────────────────────

    def graficar_trayectoria(self, paciente: str, ax=None) -> None:
        """Gráfica de la evolución de la entropía final por sesión."""
        import matplotlib.pyplot as plt

        tray = self.trayectoria(paciente)
        if not tray:
            print(f"No hay sesiones registradas para '{paciente}'.")
            return

        entropias = self.serie_temporal_entropia(paciente)
        fechas    = [e.get("timestamp", "")[:10] for e in tray if e.get("historial")]

        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.plot(range(len(entropias)), entropias, "o-",
                color="#6c63ff", lw=2, ms=8)
        ax.axhline(1.0, color="gray", ls="--", lw=1, label="Entropía máxima (1 bit)")
        ax.fill_between(range(len(entropias)), entropias,
                        alpha=0.15, color="#6c63ff")
        ax.set_xticks(range(len(fechas)))
        ax.set_xticklabels(fechas, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Entropía final de sesión (bits)")
        ax.set_title(f"Trayectoria de individuación — {paciente}")
        ax.legend()
        ax.grid(alpha=0.2)

        if standalone:
            plt.tight_layout()
            plt.show()
            plt.close()


if __name__ == "__main__":
    from experiments   import Arquetipo
    from interventions import SesionTerapeutica
    import numpy as np
    import os

    ruta_demo = "diario_demo.jsonl"
    diario    = DiarioIndividuacion(ruta_demo)

    print("Registrando 3 sesiones para 'Paciente-A'...")
    for i, alpha in enumerate([0.95, 0.85, 0.70]):
        arq    = Arquetipo(alpha, np.sqrt(max(0.0, 1 - alpha**2)))
        sesion = SesionTerapeutica(arq)
        sesion.aplicar("apertura_consciente")
        sesion.aplicar("integracion_parcial", theta=np.pi / 4)
        diario.registrar(sesion, paciente="Paciente-A", notas=f"Sesión {i + 1}")

    print(f"Pacientes en el diario: {diario.pacientes()}")

    est = diario.estadisticas_longitudinales("Paciente-A")
    print("\n=== ESTADÍSTICAS LONGITUDINALES ===")
    for k, v in est.items():
        print(f"  {k}: {v}")

    diario.graficar_trayectoria("Paciente-A")

    os.remove(ruta_demo)
