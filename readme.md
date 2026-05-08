# Laboratorio Cuántico-Junguiano en Python

**Experimentos sobre el código** que entrelazan física cuántica, psicología analítica de Carl Jung y aprendizaje automático.

Este proyecto genera simulaciones numéricas de conceptos arquetípicos usando herramientas cuánticas simples, almacena los resultados en *datasets* y entrena modelos de regresión para predecir y cuantificar relaciones entre parámetros psíquicos (represión, superposición de opuestos).

---

## 🧠 Analogía central

| Concepto cuántico   | Concepto junguiano                                            | Representación en el código                                      |
|---------------------|---------------------------------------------------------------|------------------------------------------------------------------|
| Qubit               | Par de opuestos arquetípicos (Ánima/Ánimus, Persona/Sombra)  | Dos niveles que representan la tensión consciente/inconsciente   |
| Superposición       | Ambigüedad psíquica                                           | El arquetipo no está definido hasta que se "observa" (colapso)   |
| Entrelazamiento     | Sincronicidad                                                 | Correlación entre un contenido interno y un evento externo       |
| Medición            | Toma de conciencia (insight)                                  | Colapso de la función de onda hacia un polo                      |
| Decoherencia        | Represión, censura                                            | Pérdida de la sincronicidad por "ruido" de canal                 |

---

## 🧪 Experimentos implementados

1. **Superposición de un arquetipo**  
   Un qubit con amplitudes `α|0⟩ + β|1⟩` que se mide repetidamente. Variamos `α` y registramos la frecuencia del polo Ánima (`|0⟩`).

2. **Sincronicidad por entrelazamiento**  
   Dos qubits en estado de Bell `|Φ⁺⟩`. Se introduce una probabilidad `γ` de error de fase (represión) en el primer qubit y se mide la correlación en la base X.

3. **Datasets y regresión**  
   Se generan dos conjuntos de datos:
   - **`arquetipo_prob.csv`**: relación entre `α` y la probabilidad observada de Ánima.
   - **`sincronicidad_corr.csv`**: relación entre `γ` y la correlación en base X.

   Se entrenan dos modelos sobre el dataset del arquetipo (lineal y polinomial grado 2) y uno lineal sobre el de sincronicidad. La comparación de ambos modelos del arquetipo es el experimento pedagógico central: ilustra por qué un modelo incorrecto produce error sistemático.

---

## 📁 Estructura del proyecto

```
proyecto_cuantico_jung/
├── experiments.py       # Clases con la lógica cuántica (Arquetipo, ParConDecoherencia)
├── collect_data.py      # Genera datasets variando parámetros (seed fija: 42)
├── train_regression.py  # Entrena modelos de regresión y reporta métricas train/test
├── analysis.py          # Visualizaciones: datos, predicciones y curva teórica
├── main.py              # Panel de control interactivo (menú o --auto)
├── datasets/            # CSVs generados
├── modelos/             # Modelos entrenados (.pkl)
├── exploracion/         # Prototipos descartados (solo referencia histórica)
├── requirements.txt     # Dependencias
└── README.md            # Este documento
```

---

## ⚙️ Instalación y uso

```bash
git clone https://github.com/tu-usuario/proyecto_cuantico_jung.git
cd proyecto_cuantico_jung
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Modo interactivo (recomendado):**
```bash
python main.py
```

**Modo automático (ejecuta todo el flujo sin menú):**
```bash
python main.py --auto
```

---

## 📊 Resultados esperados

**Dataset A — arquetipo:**  
La probabilidad de observar Ánima sigue `P(0) = |α|²`, una relación cuadrática. El modelo lineal produce error sistemático visible en la gráfica. El modelo polinomial de grado 2 se ajusta a la curva teórica con R² ≈ 1.0. Esta comparación es intencional: muestra que elegir el modelo equivocado tiene consecuencias medibles incluso cuando los datos son perfectamente regulares.

**Dataset B — sincronicidad:**  
La correlación en base X decrece linealmente con `γ`. El modelo lineal obtiene R² ≈ 1.0. La pérdida de sincronicidad es directamente proporcional a la intensidad de la represión.

> **Nota sobre `γ`:** el parámetro `gamma` en `aplicar_represion(gamma)` es una probabilidad de error de fase *por llamada*, no una tasa temporal acumulada. `γ = 0` deja el entrelazamiento intacto; `γ = 1` aplica el operador de Pauli Z con certeza (represión total instantánea).

> **Nota sobre el orden de medición:** `medir_base_X()` colapsa primero el qubit 1 y luego el qubit 2 sobre el estado post-colapso. Invertir el orden produce resultados estadísticamente idénticos; el orden elegido es convencional y no afecta las correlaciones medidas.

---

## 🚀 Extensiones propuestas

- Incluir más arquetipos (Persona, Sombra, Sí-mismo) como registros cuánticos adicionales.
- Simular "intervenciones terapéuticas" mediante puertas lógicas (Hadamard, rotaciones) y medir su efecto en el colapso.
- Migrar los circuitos a Qiskit para correrlos en simuladores reales o hardware cuántico.
- Entrenar modelos no lineales (árboles, redes neuronales) y comparar su capacidad predictiva.
- Crear una interfaz interactiva con Streamlit para modificar parámetros y ver predicciones en tiempo real.

---

## 📄 Licencia

MIT. Consulta el archivo `LICENSE` para más detalles.

---

## 📚 Referencias

- Jung, C. G. *Sincronicidad: un principio de conexión acausal.*
- Nielsen, M. A. & Chuang, I. L. *Quantum Computation and Quantum Information.*
- Documentación de scikit-learn, NumPy y Matplotlib.

> *"En el código, el observador no solo colapsa la función de onda; también interpreta el resultado."*
