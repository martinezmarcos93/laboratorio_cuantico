#!/usr/bin/env python3
"""
main.py — Panel de control del Laboratorio Cuántico-Junguiano v3.0.
"""

import os
import sys
import random

# Asegurar que la raíz del proyecto esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ml.collect_data       import (generar_dataset_arquetipo,
                                       generar_dataset_sincronicidad,
                                       generar_dataset_fidelidad)
    from ml.train_regression   import (entrenar_modelos_arquetipo,
                                       entrenar_modelo_sincronicidad,
                                       comparar_todos)
    from ml.analysis           import analizar_ambos
    from core.archetypes       import RegistroCuantico
    from core.experiments      import Arquetipo
    from core.interventions    import SesionTerapeutica
except ImportError as exc:
    print(f"[ERROR] No se pudo importar un módulo requerido: {exc}")
    print("Asegurate de que todos los paquetes estén en el mismo directorio raíz.")
    sys.exit(1)


def crear_directorios() -> None:
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("modelos",  exist_ok=True)


# ─── opciones del menú principal ─────────────────────────

def generar_datasets() -> None:
    print("\n🔮 Generando datasets A y B...")
    generar_dataset_arquetipo()
    generar_dataset_sincronicidad()
    print("✅ Datasets A y B creados en 'datasets/'.\n")


def generar_dataset_c() -> None:
    print("\n🔮 Generando dataset C (fidelidad arquetípica)...")
    generar_dataset_fidelidad()
    print("✅ Dataset C creado en 'datasets/'.\n")


def entrenar_modelos() -> None:
    print("\n🧠 Entrenando modelos de regresión...")
    entrenar_modelos_arquetipo()
    entrenar_modelo_sincronicidad()
    print("✅ Modelos guardados en 'modelos/'.\n")


def mostrar_analisis() -> None:
    print("\n📊 Mostrando visualizaciones...")
    analizar_ambos()
    print("✅ Gráficas cerradas.\n")


def mostrar_comparacion() -> None:
    print("\n📋 Tabla resumen de modelos:")
    tabla = comparar_todos()
    if tabla.empty:
        print("  No hay modelos entrenados aún. Ejecuta las opciones 1 y 2 primero.")
    else:
        print(tabla.to_string(index=False))
    print()


def mostrar_narrativa() -> None:
    seed = random.randint(0, 9999)
    reg  = RegistroCuantico(seed=seed)
    print(f"\n🧬 Narrativa del RegistroCuántico (seed={seed}):")
    print(f"  {reg.narrativa_estado()}")
    print(f"  Índice de individuación: {reg.indice_individuacion():.4f}\n")


def mostrar_diagnostico() -> None:
    """Diagnóstico Arquetipal Bayesiano de un RegistroCuantico."""
    try:
        from analytics.diagnostico import diagnosticar_registro, convergencia_bayesiana
    except ImportError:
        print("❌ diagnostico.py no disponible. Verificá la instalación de scipy.\n")
        return

    seed = random.randint(0, 9999)
    reg  = RegistroCuantico(seed=seed)
    print(f"\n🔬 Diagnóstico Bayesiano (seed={seed}, 200 obs/componente):")
    resultados = diagnosticar_registro(reg, n_obs=200)
    for nombre, res in resultados.items():
        print(f"  {nombre:16s} α_MAP={res['alpha_MAP']:.4f}  "
              f"IC95=[{res['IC_95_alpha'][0]}, {res['IC_95_alpha'][1]}]")

    print("\n  (Mostrando gráfica de convergencia para el Yo...)")
    yo_qubit = reg.qubits[0]
    convergencia_bayesiana(yo_qubit.alpha, n_max=300)
    print()


def mostrar_qst() -> None:
    """Tomografía de Estado Cuántico de un arquetipo."""
    import numpy as np
    from analytics.qst import (
        tomografia_bloch, medir_base_x, medir_base_y,
        reconstruir_arquetipo, graficar_esfera_bloch_2d,
    )

    alpha = float(input("  Ingresá α del arquetipo a reconstruir (0.0 – 1.0): ").strip())
    alpha = max(0.01, min(0.99, alpha))
    beta  = float(np.sqrt(1.0 - alpha**2))
    arq   = Arquetipo(alpha, beta, seed=42)

    print(f"  Simulando 300 mediciones en 3 bases para α={alpha:.3f}...")
    obs_z = [arq.medir() for _ in range(300)]
    obs_x = medir_base_x(arq, 300, seed=7)
    obs_y = medir_base_y(arq, 300, seed=13)

    res   = tomografia_bloch(obs_z, obs_x, obs_y)
    arq_r = reconstruir_arquetipo(res)

    print(f"\n  Arquetipo original:     α={alpha:.4f}")
    print(f"  Arquetipo reconstruido: α={arq_r.alpha:.4f}")
    print(f"  Vector de Bloch:  rx={res['rx']:+.4f}  rz={res['rz']:+.4f}")
    print(f"  Pureza: {res['pureza']:.4f}  (1.0 = estado puro)\n")

    rz_real = 2.0 * arq.prob_anima() - 1.0
    rx_real = 2.0 * arq.alpha * arq.beta
    graficar_esfera_bloch_2d(
        [{"rx": res["rx"], "rz": res["rz"]},
         {"rx": rx_real,   "rz": rz_real}],
        etiquetas=["QST reconstruido", "Estado real"],
    )


def mostrar_lindblad() -> None:
    """Comparación de canales de represión Lindblad."""
    try:
        from core.lindblad import comparar_canales
    except ImportError:
        print("❌ lindblad.py no disponible.\n")
        return
    print("\n⚗️  Comparando canales de represión (T1, T2 y mixto)...")
    comparar_canales()
    print()


def ejecutar_todo() -> None:
    generar_datasets()
    generar_dataset_c()
    entrenar_modelos()
    mostrar_analisis()
    mostrar_comparacion()
    print("🏁 Proceso completo finalizado.")


# ─── menú ────────────────────────────────────────────────

def menu() -> None:
    crear_directorios()
    opciones = {
        "1":  ("Generar datasets A y B",                    generar_datasets),
        "2":  ("Entrenar modelos de regresión",             entrenar_modelos),
        "3":  ("Mostrar análisis (gráficas)",               mostrar_analisis),
        "4":  ("Ejecutar todo el flujo",                    ejecutar_todo),
        "5":  ("Generar dataset C (fidelidad)",             generar_dataset_c),
        "6":  ("Comparar todos los modelos (tabla)",        mostrar_comparacion),
        "7":  ("Narrativa psíquica (RegistroCuantico)",     mostrar_narrativa),
        "8":  ("Diagnóstico Arquetipal Bayesiano",          mostrar_diagnostico),
        "9":  ("Tomografía de Estado Cuántico (QST)",       mostrar_qst),
        "10": ("Canal de Lindblad — comparar represiones",  mostrar_lindblad),
        "11": ("Salir",                                     None),
    }
    salida = max(opciones.keys())
    while True:
        print("=" * 60)
        print("  🧬 LABORATORIO CUÁNTICO-JUNGUIANO  v3.0")
        print("=" * 60)
        for key, (label, _) in opciones.items():
            print(f"  {key:>2}. {label}")
        opcion = input(f"\nElegí una opción (1-{salida}): ").strip()
        if opcion == salida:
            print("Saliendo. ¡Hasta la próxima sincronicidad!")
            break
        elif opcion in opciones:
            opciones[opcion][1]()
        else:
            print("❌ Opción no válida.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        crear_directorios()
        ejecutar_todo()
    else:
        menu()
