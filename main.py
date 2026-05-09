#!/usr/bin/env python3
"""
main.py — Panel de control del Laboratorio Cuántico-Junguiano.

BUGS CORREGIDOS:
  1. [MENOR] Los imports estaban dentro de funciones (opciones del menú),
     lo que retrasaba la detección de errores de importación al momento de
     uso. Movidos al tope con manejo de ImportError descriptivo.

MEJORAS AÑADIDAS:
  - Opción 5: Generar dataset de fidelidad arquetípica (Dataset C).
  - Opción 6: Comparación de modelos (tabla resumen en terminal).
  - Opción 7: Narrativa de estado de un RegistroCuantico aleatorio.
  - --auto ejecuta el flujo completo incluyendo Dataset C.
"""

import os
import sys

# BUG FIX 1: imports al tope — errores detectados al iniciar
try:
    from collect_data    import (generar_dataset_arquetipo,
                                 generar_dataset_sincronicidad,
                                 generar_dataset_fidelidad)
    from train_regression import (entrenar_modelos_arquetipo,
                                  entrenar_modelo_sincronicidad,
                                  comparar_todos)
    from analysis        import analizar_ambos
except ImportError as exc:
    print(f"[ERROR] No se pudo importar un módulo requerido: {exc}")
    print("Asegurate de que todos los archivos .py estén en el mismo directorio.")
    sys.exit(1)


def crear_directorios() -> None:
    """Asegura que existan las carpetas necesarias."""
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("modelos",  exist_ok=True)


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
    from archetypes import RegistroCuantico
    import random
    seed = random.randint(0, 9999)
    reg  = RegistroCuantico(seed=seed)
    print(f"\n🧬 Narrativa del RegistroCuántico (seed={seed}):")
    print(f"  {reg.narrativa_estado()}")
    print(f"  Índice de individuación: {reg.indice_individuacion():.4f}\n")


def ejecutar_todo() -> None:
    generar_datasets()
    generar_dataset_c()
    entrenar_modelos()
    mostrar_analisis()
    mostrar_comparacion()
    print("🏁 Proceso completo finalizado.")


def menu() -> None:
    """Menú interactivo."""
    crear_directorios()
    opciones = {
        "1": ("Generar datasets A y B (experimentos)",  generar_datasets),
        "2": ("Entrenar modelos de regresión",          entrenar_modelos),
        "3": ("Mostrar análisis (gráficas)",            mostrar_analisis),
        "4": ("Ejecutar todo el flujo",                 ejecutar_todo),
        "5": ("Generar dataset C (fidelidad)",          generar_dataset_c),
        "6": ("Comparar todos los modelos (tabla)",     mostrar_comparacion),
        "7": ("Narrativa psíquica de registro cuántico",mostrar_narrativa),
        "8": ("Salir",                                  None),
    }
    while True:
        print("=" * 55)
        print("  🧬 LABORATORIO CUÁNTICO-JUNGUIANO  v2.0")
        print("=" * 55)
        for key, (label, _) in opciones.items():
            print(f"  {key}. {label}")
        opcion = input("\nElegí una opción (1-8): ").strip()
        if opcion == "8":
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
