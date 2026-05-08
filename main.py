#!/usr/bin/env python3
"""
Panel de control del Laboratorio Cuántico-Junguiano.
Permite ejecutar los experimentos, entrenar modelos y visualizar resultados.
"""
import os
import sys

# --- Imports al tope: los errores se detectan al iniciar, no al ejecutar ---
from collect_data import generar_dataset_arquetipo, generar_dataset_sincronicidad
from train_regression import entrenar_arquetipo, entrenar_sincronicidad
from analysis import analizar_ambos


def crear_directorios():
    """Asegura que existan las carpetas necesarias."""
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("modelos",  exist_ok=True)


def generar_datasets():
    """Ejecuta la recolección de datos (experimentos)."""
    print("\n🔮 Generando datasets...")
    generar_dataset_arquetipo()
    generar_dataset_sincronicidad()
    print("✅ Datasets creados en 'datasets/'.\n")


def entrenar_modelos():
    """Entrena regresión lineal y polinomial para el arquetipo, lineal para sincronicidad."""
    print("\n🧠 Entrenando modelos de regresión...")
    entrenar_arquetipo()
    entrenar_sincronicidad()
    print("✅ Modelos guardados en 'modelos/'.\n")


def mostrar_analisis():
    """Genera las gráficas de análisis."""
    print("\n📊 Mostrando visualizaciones...")
    analizar_ambos()
    print("✅ Gráficas cerradas. Volviendo al menú.\n")


def ejecutar_todo():
    """Flujo completo: datasets → modelos → análisis."""
    generar_datasets()
    entrenar_modelos()
    mostrar_analisis()
    print("🏁 Proceso completo finalizado.")


def menu():
    """Menú interactivo."""
    crear_directorios()
    opciones = {
        "1": ("Generar datasets (experimentos)",      generar_datasets),
        "2": ("Entrenar modelos de regresión",         entrenar_modelos),
        "3": ("Mostrar análisis (gráficas)",           mostrar_analisis),
        "4": ("Ejecutar todo el flujo",                ejecutar_todo),
        "5": ("Salir",                                 None),
    }
    while True:
        print("=" * 50)
        print("  🧬 LABORATORIO CUÁNTICO-JUNGUIANO")
        print("=" * 50)
        for key, (label, _) in opciones.items():
            print(f"{key}. {label}")
        opcion = input("Elige una opción (1-5): ").strip()
        if opcion == "5":
            print("Saliendo del laboratorio. ¡Hasta la próxima sincronicidad!")
            break
        elif opcion in opciones:
            opciones[opcion][1]()
        else:
            print("❌ Opción no válida. Inténtalo de nuevo.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        crear_directorios()
        ejecutar_todo()
    else:
        menu()
