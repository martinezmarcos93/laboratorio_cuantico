# Laboratorio Cuántico-Junguiano

**Experimentos computacionales** que entrelazan física cuántica, psicología analítica de Carl Jung y aprendizaje automático.

El proyecto genera simulaciones numéricas de conceptos arquetípicos usando herramientas cuánticas simplificadas, almacena los resultados en datasets y entrena modelos de regresión para predecir y cuantificar relaciones entre parámetros psíquicos (represión, superposición de opuestos). Incluye un dashboard interactivo en Streamlit con sliders en tiempo real y un pipeline de intervenciones terapéuticas modeladas como puertas cuánticas.

---

## Analogía central

| Concepto cuántico | Concepto junguiano | Representación en el código |
|---|---|---|
| Qubit | Par de opuestos arquetípicos (Ánima/Ánimus) | Dos niveles que representan la tensión consciente/inconsciente |
| Superposición | Ambigüedad psíquica | El arquetipo no está definido hasta que se "observa" (colapso) |
| Entrelazamiento | Sincronicidad | Correlación entre un contenido interno y un evento externo |
| Medición | Toma de conciencia (insight) | Colapso de la función de onda hacia un polo |
| Decoherencia | Represión, censura | Pérdida de la sincronicidad por canal de desfase |
| Puerta Hadamard | Apertura consciente | Lleva el estado a superposición máxima |
| Puerta Ry(θ) | Integración terapéutica | Rotación gradual entre polos |
| Pauli-Z | Represión de fase | Degrada el entrelazamiento sin invertir el polo |
| Pauli-X | Proyección psíquica | Inversión total del polo dominante |

---

## Experimentos implementados

**Experimento A — Superposición de un arquetipo**

Un qubit `α|0⟩ + β|1⟩` se mide repetidamente variando `α`. La frecuencia de colapso al polo Ánima sigue `P(0) = |α|²` (relación cuadrática). Se comparan un modelo lineal y uno polinomial para ilustrar el costo de elegir el modelo equivocado.

**Experimento B — Sincronicidad por entrelazamiento**

Dos qubits en estado de Bell `|Φ⁺⟩`. Se introduce una probabilidad `γ` de error de fase (represión) y se mide la correlación en base X. La relación teórica es `correlación = 1 − γ`, perfectamente lineal. El modelo lineal obtiene R² ≈ 1.0.

**Experimento C — Sesión terapéutica (pipeline de puertas)**

Secuencia de intervenciones cuánticas sobre un arquetipo inicial. Cada puerta transforma el estado y se registra en el historial. Disponible en el dashboard Streamlit.

---

## Estructura del proyecto

```
laboratorio_cuantico/
├── datasets/
├── docs/
│   └── guia-de-usuario.md
├── exploracion/
│   ├── sincronicidad_mediante_entrelazamiento.py
│   └── superposición_arquetípica.py
├── modelos/
│   ├── regresion_arquetipo_lineal.pkl
│   ├── regresion_arquetipo_poli.pkl
│   └── regresion_sincronicidad.pkl
├── analysis.py
├── archetypes.py
├── collect_data.py
├── experiments.py
├── interventions.py
├── main.py
├── readme.md
├── requirements.txt
├── streamlit_app.py
└── train_regression.py
```

---

## Instalación y uso

```bash
git clone https://github.com/tu-usuario/proyecto_cuantico_jung.git
cd proyecto_cuantico_jung
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Modo interactivo:**
```bash
python main.py
```

**Modo automático:**
```bash
python main.py --auto
```

**Dashboard Streamlit:**
```bash
streamlit run streamlit_app.py
```

---

## Resultados esperados

**Dataset A — Arquetipo:**
El modelo lineal produce error sistemático visible en la gráfica. El polinomial de grado 2 se ajusta a la curva teórica `P(0) = α²` con R² ≈ 1.0.

**Dataset B — Sincronicidad:**
La correlación decrece linealmente con `γ`. El modelo lineal obtiene R² ≈ 1.0.

> **Nota sobre `γ`:** con `γ = 0` el entrelazamiento es intacto (correlación = 1). Con `γ = 1` se aplica Pauli-Z con certeza, produciendo **anticorrelación perfecta** (correlación = 0 en la métrica P(x1=x2)), no estado mixto.

---

## Documentación

Ver `docs/guia-de-usuario.md` para una explicación completa de qué hace cada experimento, qué significa cada parámetro y cómo interpretar las gráficas desde la física cuántica y la psicología junguiana.

---

## Extensiones propuestas

- Tomografía del estado cuántico para reconstruir la matriz de densidad del arquetipo
- Entrelazamiento multi-arquetipo con grafo de sincronicidad (NetworkX + Qiskit)
- Canal de Lindblad generalizado con cuatro mecanismos de represión distintos
- Diagnóstico arquetipal bayesiano: inferir `α` a partir de observaciones conductuales
- Event Sourcing para registrar y reproducir el proceso de individuación

---

## Dependencias

```
numpy >= 1.21
matplotlib >= 3.5
pandas >= 1.3
scikit-learn >= 1.0
joblib >= 1.1
streamlit >= 1.28      # solo para streamlit_app.py
```

---

## Referencias

- Jung, C. G. *Sincronicidad: un principio de conexión acausal.*
- Nielsen, M. A. & Chuang, I. L. *Quantum Computation and Quantum Information.*
- Documentación de scikit-learn, NumPy y Matplotlib.

---

## Licencia

MIT.
