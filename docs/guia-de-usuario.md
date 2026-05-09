# Guía de Usuario — Laboratorio Cuántico-Junguiano

> *"En el código, el observador no solo colapsa la función de onda; también interpreta el resultado."*

Esta guía te explica qué podés hacer con el proyecto, qué significa cada parte y cómo interpretarlo, tanto desde la física cuántica como desde la psicología de Jung.

---

## ¿De qué trata este proyecto?

El Laboratorio Cuántico-Junguiano es un entorno de simulación que usa mecánica cuántica simplificada para modelar conceptos de la psicología analítica de Carl Jung. No es un simulador clínico ni un simulador cuántico de hardware real — es una herramienta pedagógica y exploratoria que hace visible, de forma computacional y matemática, cómo funcionan ciertas ideas psicológicas cuando se traducen al lenguaje de la física cuántica.

**Lo que podés hacer concretamente:**

- Simular cómo un arquetipo psíquico "colapsa" hacia un polo cuando se lo observa
- Medir cómo la represión destruye la sincronicidad entre eventos psíquicos
- Entrenar modelos de regresión que predicen esas relaciones
- Visualizar por qué un modelo matemático incorrecto falla incluso con datos perfectos
- Explorar intervenciones terapéuticas como puertas cuánticas
- Analizar la psique completa como un registro de 5 qubits

---

## Modos de uso

### Modo interactivo (menú en consola)

```bash
python main.py
```

Muestra un menú con 4 operaciones que podés ejecutar por separado o en secuencia:

```
1. Generar datasets (experimentos)
2. Entrenar modelos de regresión
3. Mostrar análisis (gráficas)
4. Ejecutar todo el flujo
```

Usá este modo cuando quieras controlar paso a paso qué corre, o cuando estés explorando el código.

### Modo automático

```bash
python main.py --auto
```

Ejecuta el flujo completo sin preguntar: genera datos → entrena modelos → muestra gráficas. Útil para reproducir resultados rápidamente o en scripts.

### Dashboard interactivo (Streamlit)

```bash
streamlit run streamlit_app.py
```

Abre una aplicación web en tu navegador con sliders, gráficas en tiempo real y un pipeline de intervenciones terapéuticas interactivo. Es la forma más visual de explorar el proyecto.

---

## Los dos experimentos centrales

### Experimento A — Superposición del Arquetipo

**¿Qué simula?**

Un qubit con amplitudes `α|0⟩ + β|1⟩` que se mide muchas veces. Se varía `α` y se registra con qué frecuencia el qubit colapsa al polo `|0⟩` (llamado "Ánima").

**¿Qué significa físicamente?**

Un qubit en superposición no tiene un valor definido hasta que se lo mide. Al medirlo, colapsa con probabilidad `P(0) = |α|²`. Esta relación es cuadrática, no lineal.

**¿Qué significa junguianamente?**

El qubit representa un par de opuestos arquetípicos (por ejemplo, Ánima y Ánimus — los polos femenino y masculino en la psique de Jung). Mientras el arquetipo no es confrontado conscientemente, existe en superposición — sin polo dominante definido. El momento de la medición es el "insight": el arquetipo colapsa hacia un polo, volviéndose conscientemente reconocible.

**¿Qué muestra la gráfica?**

Tres curvas sobre el mismo eje:
- **Puntos dispersos** (azul): datos simulados con ruido estadístico
- **Línea roja punteada**: predicción del modelo lineal
- **Línea verde**: predicción del modelo polinomial de grado 2
- **Línea negra punteada**: la curva teórica exacta `P(0) = α²`

El modelo lineal se desvía sistemáticamente. El polinomial coincide casi perfectamente con la curva teórica. Esto es intencional: ilustra que elegir el modelo matemático incorrecto produce error estructural, no solo ruido.

**Parámetros del dataset (`arquetipo_prob.csv`):**

| Columna | Descripción |
|---------|-------------|
| `alpha` | Amplitud del polo Ánima. Varía de 0 a 1 en pasos de 0.05. |
| `prob_anima` | Proporción observada de colapsos hacia Ánima en 500 mediciones. |

---

### Experimento B — Sincronicidad por entrelazamiento

**¿Qué simula?**

Dos qubits preparados en el estado de Bell `|Φ⁺⟩ = (|00⟩ + |11⟩)/√2`. Se introduce un canal de desfase con intensidad `γ` sobre el primer qubit y se mide la correlación entre ambos en la base X.

**¿Qué significa físicamente?**

El estado de Bell es el estado cuántico más entrelazado posible: medir uno de los qubits determina instantáneamente el resultado del otro. El canal de desfase (con parámetro `γ`) degrada ese entrelazamiento mediante un error de fase aleatorio. Con `γ = 0`, la correlación es perfecta. Con `γ = 1`, se aplica el operador de Pauli Z con certeza, invirtiendo la correlación a su opuesto.

**¿Qué significa junguianamente?**

El entrelazamiento modela la **sincronicidad** de Jung: la correlación acausal entre un evento interno (pensamiento, sueño, imagen) y un evento externo. El parámetro `γ` modela la **represión**: cuanto más reprimido está un contenido psíquico, más se degrada la correlación entre el mundo interno y el externo, y la sincronicidad deja de manifestarse.

Con `γ = 1`, el operador Z transforma `|Φ⁺⟩` en `|Φ⁻⟩`, que tiene anticorrelación perfecta — no ausencia de correlación. Esto modela la represión total, donde el contenido inconsciente se manifiesta de forma completamente invertida respecto a su expresión consciente (proyección pura).

**¿Qué muestra la gráfica?**

- **Puntos dispersos** (azul): correlaciones estimadas con ruido de muestreo
- **Línea roja punteada**: predicción del modelo lineal
- **Línea negra punteada**: curva teórica exacta `correlación = 1 − γ`

La relación es perfectamente lineal. El modelo lineal obtiene R² ≈ 1.0.

**Parámetros del dataset (`sincronicidad_corr.csv`):**

| Columna | Descripción |
|---------|-------------|
| `gamma` | Intensidad de la represión (probabilidad de desfase). Varía de 0 a 1. |
| `correlacion` | Proporción de mediciones donde los dos qubits coincidieron en base X. |

---

## Los modelos de regresión

El proyecto entrena tres modelos de scikit-learn y los guarda en `modelos/`:

| Archivo | Tipo | Dataset | R² esperado |
|---------|------|---------|-------------|
| `regresion_arquetipo_lineal.pkl` | Regresión lineal | Arquetipo | Bajo (~0.75) |
| `regresion_arquetipo_poli.pkl` | Polinomial grado 2 | Arquetipo | ~1.0 |
| `regresion_sincronicidad.pkl` | Regresión lineal | Sincronicidad | ~1.0 |

**¿Por qué el modelo lineal del arquetipo tiene R² bajo si los datos son perfectamente regulares?**

Porque la relación real es cuadrática (`P(0) = α²`), y un modelo lineal no puede capturar una curva. Aunque los datos no tengan ruido extra, el modelo equivocado produce error sistemático — siempre sobreestima para `α` bajos y altos, y subestima en el centro.

Esto es una lección central del proyecto: el error no siempre viene de datos ruidosos. A veces viene de supuestos incorrectos sobre la forma de la relación.

---

## Los arquetipos junguianos como qubits

El módulo `archetypes.py` modela los 5 componentes clásicos de la psique:

| Arquetipo | P(Ánima) inicial | Entropía | Significado junguiano |
|-----------|-----------------|----------|----------------------|
| **Yo (Ego)** | 0.60 | ~0.97 bits | Núcleo de la consciencia. Levemente sesgado al polo activo pero no rígido. |
| **Persona** | 0.85 | ~0.61 bits | Máscara social. Sesgada hacia el polo "presentable". Poca ambigüedad. |
| **Sombra** | 0.15 | ~0.61 bits | Contenidos reprimidos. Sesgada al polo opuesto al Yo. |
| **Ánima/Ánimus** | 0.50 | 1.00 bit | Contrasexual interno. Superposición perfecta: máxima ambigüedad. |
| **Sí-mismo** | 0.50 | 1.00 bit | Totalidad psíquica. Centro que integra todos los opuestos. |

**¿Qué es la entropía en este contexto?**

La entropía de von Neumann (en bits) mide la ambigüedad del estado psíquico. Entropía 0 significa que el arquetipo está completamente polarizado — ya no hay tensión, ya no hay superposición. Entropía 1 bit es la máxima tensión: el arquetipo puede colapsar en cualquier dirección con igual probabilidad.

Junguianamente, entropía alta no es algo "malo" — es el estado de máximo potencial, donde el proceso de individuación todavía tiene margen de movimiento.

**¿Qué es la tensión Yo↔Sombra?**

Es `|P(Ánima|Yo) - P(Ánimus|Sombra)|`. Mide qué tan "complementarios" son estos dos arquetipos. Una tensión alta indica que la Sombra está fuertemente polarizada en sentido opuesto al Yo — contenidos muy reprimidos. La individuación junguiana trabaja para reducir esta tensión mediante la integración consciente de la Sombra.

---

## Las intervenciones terapéuticas

El módulo `interventions.py` implementa puertas cuánticas como transformaciones sobre el estado de un arquetipo. Cada una tiene un correlato junguiano:

### `apertura_consciente` — Puerta Hadamard

Lleva cualquier estado a superposición perfecta `P(Ánima) = 0.5`.

**Jung:** El paciente suspende todo juicio previo y se abre por completo al inconsciente. No importa desde dónde empiece — el resultado es siempre máxima ambigüedad, máxima apertura.

### `integracion_parcial` — Rotación Ry(θ)

Desplaza gradualmente el equilibrio entre polos según el ángulo `θ`.

- `θ = 0`: sin cambio
- `θ = π/2`: rotación de 45°, acerca al equilibrio
- `θ = π`: inversión completa (equivalente a proyección)

**Jung:** El proceso de integración gradual de la Sombra. No ocurre de golpe — cada sesión terapéutica desplaza el estado un ángulo, acercándolo a la integración sin forzar una crisis.

### `amplificacion` — Rotación dirigida

Refuerza un polo específico (Ánima o Ánimus) con intensidad controlada.

**Jung:** La técnica de amplificación junguiana: enfatizar deliberadamente un contenido psíquico específico para hacerlo más consciente y disponible para el trabajo analítico.

### `proyeccion_anima` / `proyeccion_animus`

Amplificación orientada hacia uno de los dos polos.

**Jung:** La proyección clásica — atribuir al mundo externo (o a otra persona) un contenido que pertenece al mundo interno. La amplificación en la dirección correcta puede revertirla.

---

## El pipeline terapéutico (Streamlit)

En la sección "💊 Sesión terapéutica" podés:

1. Configurar el `α` inicial del arquetipo (por ejemplo, `α = 0.95` representa un Yo muy dominante sobre el Ánimus)
2. Aplicar intervenciones en secuencia
3. Ver la tabla de evolución del estado paso a paso
4. Ver la gráfica de P(Ánima) y P(Ánimus) a lo largo del proceso

**¿Cómo leer la gráfica de sesión?**

- Una curva que se aplana cerca de 0.5 indica un estado de alta apertura (superposición activa)
- Una curva que sube o baja bruscamente indica una intervención de alta intensidad
- La línea punteada en 0.5 es el eje de equilibrio — el Sí-mismo junguiano

---

## Estructura de archivos

```
proyecto/
├── experiments.py       # Física cuántica: Arquetipo y ParConDecoherencia
├── archetypes.py        # Arquetipos junguianos extendidos (5 componentes)
├── interventions.py     # Puertas cuánticas como intervenciones terapéuticas
├── collect_data.py      # Genera los datasets variando parámetros
├── train_regression.py  # Entrena y evalúa los modelos de regresión
├── analysis.py          # Gráficas comparativas de modelos
├── main.py              # Menú interactivo y modo --auto
├── streamlit_app.py     # Dashboard web completo
├── datasets/            # CSVs generados (se crean al correr el proyecto)
├── modelos/             # Modelos .pkl entrenados (se crean al entrenar)
├── docs/                # Esta documentación
└── requirements.txt     # Dependencias Python
```

---

## Flujo completo recomendado

Si es la primera vez que usás el proyecto:

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Correr el flujo completo
python main.py --auto

# 3. Explorar con el dashboard
streamlit run streamlit_app.py
```

Si solo querés explorar la física sin entrenar modelos, podés importar directamente desde Python:

```python
import numpy as np
from experiments import Arquetipo, ParConDecoherencia
from archetypes import RegistroCuantico
from interventions import SesionTerapeutica

# Crear un arquetipo con dominancia de Ánima
arq = Arquetipo(alpha=0.9, beta=np.sqrt(1 - 0.81))
print(f"P(Ánima) = {arq.prob_anima():.4f}")
print(f"Entropía = {arq.entropia_de_von_neumann():.4f} bits")

# Simular 10 "insights"
resultados = [arq.medir() for _ in range(10)]
print(f"Colapsos: {resultados}")  # 0 = Ánima, 1 = Ánimus

# Pipeline terapéutico
sesion = SesionTerapeutica(arq)
sesion.aplicar("apertura_consciente")
sesion.aplicar("integracion_parcial", theta=np.pi / 4)
print(sesion.resumen())
```

---

## Glosario rápido

| Término | Física cuántica | Jung |
|---------|----------------|------|
| **Qubit** | Sistema de 2 niveles con amplitudes complejas | Par de opuestos arquetípicos |
| **Superposición** | Estado no definido antes de la medición | Ambigüedad psíquica |
| **Colapso** | La medición fuerza un valor definido | Insight: el arquetipo se vuelve consciente |
| **Entrelazamiento** | Correlación no local entre qubits | Sincronicidad: conexión acausal |
| **Decoherencia** | Pérdida de correlaciones cuánticas por interacción con el entorno | Represión, censura psíquica |
| **Alpha (α)** | Amplitud del estado `\|0⟩` | Fuerza del polo Ánima |
| **Beta (β)** | Amplitud del estado `\|1⟩` | Fuerza del polo Ánimus |
| **Gamma (γ)** | Probabilidad de error de fase en el canal de desfase | Intensidad de la represión |
| **Puerta Hadamard** | Crea superposición perfecta | Apertura total al inconsciente |
| **Puerta Ry(θ)** | Rotación en el plano de Bloch | Intervención terapéutica gradual |
| **Pauli-Z** | Inversión de fase | Represión de contenido psíquico |
| **Pauli-X** | Inversión de polo | Proyección psíquica total |
| **Entropía de Shannon** | Incertidumbre del estado en bits | Grado de ambigüedad psíquica activa |
| **R²** | Coeficiente de determinación del modelo | Qué tan bien captura el modelo la relación real |

---

*Para continuar con la referencia técnica de cada módulo, ver `docs/referencia-tecnica.md`.*
