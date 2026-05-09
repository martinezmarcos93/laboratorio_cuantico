# Informe de Análisis — Laboratorio Cuántico-Junguiano v2.0

> Análisis realizado desde triple experticia: física cuántica, psicología junguiana y arquitectura de software.

---

## PARTE I — BUGS CORREGIDOS

### 🔴 CRÍTICOS (habrían causado resultados silenciosamente incorrectos)

#### BUG-01 | `experiments.py` — Contaminación del RNG global (afecta a TODOS los archivos)

**Archivo:** `experiments.py`, `collect_data.py`
**Síntoma:** `np.random.seed(SEED)` y `np.random.seed(seed)` en `__init__` mutan el estado
global del generador de números aleatorios de NumPy. Cualquier módulo importado *después*
de instanciar un `Arquetipo` o `ParConDecoherencia` hereda ese estado.

**Consecuencia concreta:** los resultados de `collect_data.py` cambiaban si se importaba
otro módulo antes, haciendo los datasets **no reproducibles** en pipelines reales.

**Corrección aplicada en todos los archivos:**
```python
# ANTES (bug):
np.random.seed(seed)
...
return 0 if np.random.random() < abs(self.alpha)**2 else 1

# DESPUÉS (correcto):
self._rng = np.random.default_rng(seed)   # RNG local, no global
...
return 0 if self._rng.random() < abs(self.alpha)**2 else 1
```

**Lectura arquetípica:** este bug es la "Sombra" del diseño — un efecto oculto que
actúa sobre el inconsciente colectivo del programa (el estado global), contaminando
todos los subsistemas sin que ninguno lo sepa conscientemente.

---

#### BUG-02 | `streamlit_app.py` — Doble llamada a `medir_base_X()` (ya documentado, confirmado crítico)

**Síntoma original:**
```python
# Código original — DOS llamadas independientes, no dos qubits del mismo par:
if par.medir_base_X()[0] == par.medir_base_X()[1]
```
Esto comparaba el qubit-1 de una medición con el qubit-2 de una medición *diferente*,
produciendo correlación ≈ 0.5 para cualquier valor de γ. El canal de desfase quedaba
completamente invisible.

**Corrección (confirmada en la versión que subiste):**
```python
x1, x2 = par.medir_base_X()
if x1 == x2: iguales += 1
```

**Lectura cuántica:** medir dos veces colapsa el estado dos veces — en mecánica cuántica
real, la segunda medición actúa sobre un estado post-colapso diferente. El bug
recreaba exactamente ese error conceptual en el software.

---

#### BUG-03 | `experiments.py` — `aplicar_rotacion` genera Arquetipo sin `_rng`

**Síntoma:** `object.__new__(Arquetipo)` crea la instancia saltando `__init__`,
por lo que el objeto resultante no tiene `self._rng`. Al llamar `.medir()` en
ese objeto: `AttributeError: 'Arquetipo' object has no attribute '_rng'`.

**Corrección:**
```python
obj._rng = np.random.default_rng()   # RNG propio para el estado hijo
```

**Afectados:** `interventions.py` (todas las puertas), `SesionTerapeutica.aplicar()`.

---

#### BUG-04 | `collect_data.py` — `np.random.seed(SEED)` a nivel de módulo

**Síntoma:** la seed global se establece en el momento del `import collect_data`,
no en el momento de llamar las funciones. Si otros módulos se importan entre medio,
el estado del RNG ya no es el esperado.

**Corrección:** `rng = np.random.default_rng(SEED)` dentro de cada función.

---

### 🟡 MENORES (comportamiento incorrecto o frágil, no silencioso)

| # | Archivo | Descripción | Corrección |
|---|---------|-------------|-----------|
| M-01 | `archetypes.py` | `AnimaAnimus(1.0, 1.0)` depende de normalización implícita — correcto pero confuso | Cambiado a `1/√2` explícito |
| M-02 | `archetypes.py` | `indice_individuacion` con `abs(h_media - 0.5) * 2` producía valores fuera de [0,1] para h_media > 1 si se cambiaba la escala | Fórmula reescrita como promedio ponderado monotónico |
| M-03 | `train_regression.py` | Sin guard de existencia de CSV antes de `pd.read_csv()` → `FileNotFoundError` sin contexto | `_cargar_dataset()` con mensaje accionable |
| M-04 | `analysis.py` | Sin guard de existencia de `.pkl` antes de `joblib.load()` → error críptico | `_check_files()` con paths y sugerencia |
| M-05 | `interventions.py` | `SesionTerapeutica.aplicar` no capturaba `TypeError` de kwargs incorrectos | `try/except TypeError` con mensaje descriptivo |
| M-06 | `main.py` | Imports dentro de funciones del menú → errores de importación detectados tarde | Imports movidos al tope con `ImportError` descriptivo |

---

## PARTE II — MEJORAS E IMPLEMENTACIONES CUÁNTICO-JUNGUIANAS

### MEJORA 1 — `fidelidad()` y `distancia_traza()` en `Arquetipo`

**Inspiración cuántica:** la fidelidad F = |⟨ψ|φ⟩|² es la medida canónica de
similitud entre estados cuánticos. En el contexto junguiano, mide qué tan
"arquetípicamente resonantes" son dos estados psíquicos.

**Inspiración junguiana:** el proceso de individuación implica acercarse a un
arquetipo ideal. La fidelidad provee una métrica cuantitativa de ese proceso.

```python
# Implementado en experiments.py
def fidelidad(self, otro: "Arquetipo") -> float:
    overlap = self.alpha * np.conj(otro.alpha) + self.beta * np.conj(otro.beta)
    return float(abs(overlap) ** 2)

def distancia_traza(self, otro: "Arquetipo") -> float:
    return float(np.sqrt(max(0.0, 1 - self.fidelidad(otro))))
```

**Uso práctico en terapia:** comparar el arquetipo inicial de un paciente con
el arquetipo "SiMismo" (superposición perfecta) para cuantificar la distancia
al estado de individuación.

---

### MEJORA 2 — `grafo_sincronicidad()` en `RegistroCuantico`

**Inspiración cuántica:** la matriz de fidelidades mutuas entre los 5 componentes
de la psique es el análogo de la "matriz de solapamiento" en química cuántica.

**Inspiración junguiana:** Jung describía los arquetipos como nodos de una red
psíquica. Esta mejora lo formaliza como un grafo pesado donde los pesos son
fidelidades cuánticas.

```python
# Implementado en archetypes.py
def grafo_sincronicidad(self) -> np.ndarray:
    n   = len(self.qubits)
    mat = np.zeros((n, n))
    for i, qi in enumerate(self.qubits):
        for j, qj in enumerate(self.qubits):
            mat[i, j] = qi.fidelidad(qj)
    return mat
```

**Visualización propuesta:** NetworkX con pesos como grosor de aristas.
```python
import networkx as nx
G   = nx.Graph()
mat = reg.grafo_sincronicidad()
for i, ci in enumerate(reg.COMPONENTES):
    for j, cj in enumerate(reg.COMPONENTES):
        if i < j and mat[i,j] > 0.1:
            G.add_edge(ci, cj, weight=mat[i,j])
nx.draw_networkx(G, width=[G[u][v]['weight']*5 for u,v in G.edges()])
```

---

### MEJORA 3 — `entropia_entrelazamiento()` en `ParConDecoherencia`

**Inspiración cuántica:** la entropía de Von Neumann de la matriz de densidad
reducida (trazar sobre uno de los qubits) mide el grado de entrelazamiento
cuántico real, más allá de la simple correlación en base X.

```python
# Implementado en experiments.py
def entropia_entrelazamiento(self) -> float:
    rho_1   = np.trace(self.rho.reshape(2, 2, 2, 2), axis1=1, axis2=3)
    eigvals = np.linalg.eigvalsh(rho_1)
    eigvals = eigvals[eigvals > 1e-12]
    return float(-np.sum(eigvals * np.log2(eigvals)))
```

**Insight cuántico-junguiano:** la correlación en base X (lo que ya medías)
captura la *manifestación* de la sincronicidad. La entropía de entrelazamiento
captura su *profundidad estructural* — cuánta información comparida no-local
existe entre el evento interno y el externo. Son complementarias.

---

### MEJORA 4 — `GridSearchCV` en `entrenar_modelos_arquetipo()`

**Inspiración cuántica:** en lugar de asumir un hamiltoniano (modelo) de grado
fijo, realizar una búsqueda sobre el espacio de modelos es análogo a la
tomografía cuántica — dejar que los datos colapsen la ambigüedad del modelo.

```python
# Implementado en train_regression.py
param_grid = {"polynomialfeatures__degree": [2, 3, 4]}
pipe_base  = make_pipeline(PolynomialFeatures(include_bias=False), LinearRegression())
gs         = GridSearchCV(pipe_base, param_grid, cv=5, scoring="r2")
gs.fit(X_tr, y_tr)
poli = gs.best_estimator_
```

**Resultado esperado:** el grado 2 ganará siempre (relación teórica exacta es
cuadrática), pero el framework ahora lo *demuestra empíricamente* en lugar de
asumirlo — diferencia epistémica fundamental.

---

### MEJORA 5 — `narrativa_estado()` en `RegistroCuantico`

**Inspiración junguiana:** la psicología analítica se comunica en narrativa, no
en números. Este método traduce los índices cuantitativos (tensión, entropía,
individuación) a lenguaje clínico-simbólico.

```python
# Implementado en archetypes.py
def narrativa_estado(self) -> str:
    idx  = self.indice_individuacion()
    tens = self.tension_yo_sombra()
    ...
    return f"Estado psíquico: {estado_global}. Tensión Yo↔Sombra: {tens:.3f}. ..."
```

**Extensión propuesta:** conectar esta narrativa a un LLM vía API para generar
informes clínicos completos:
```python
import anthropic
client   = anthropic.Anthropic()
narrativa = reg.narrativa_estado()
prompt   = f"Como psicólogo analítico, interpreta este estado: {narrativa}"
resp     = client.messages.create(model="claude-sonnet-4-20250514",
                                  max_tokens=500,
                                  messages=[{"role":"user","content":prompt}])
print(resp.content[0].text)
```

---

### MEJORA 6 (PROPUESTA ARQUITECTÓNICA) — Event Sourcing para la Individuación

**Inspiración cuántica:** en mecánica cuántica, la historia de mediciones importa
(el estado post-colapso depende del resultado). El proceso terapéutico es
inherentemente *path-dependent*.

**Inspiración junguiana:** Jung enfatizaba el análisis longitudinal del proceso
de individuación. El historial de sesiones es tan importante como el estado actual.

**Implementación propuesta:**
```python
# events.py — nuevo módulo
from dataclasses import dataclass, field
from datetime import datetime
import json, pathlib

@dataclass
class EventoTerapeutico:
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tipo: str = ""
    parametros: dict = field(default_factory=dict)
    estado_post: dict = field(default_factory=dict)

class DiarioIndividuacion:
    """
    Event Store para el proceso de individuación de un paciente.
    Permite replay completo de la trayectoria psíquica.
    """
    def __init__(self, path: str = "diario.jsonl"):
        self.path = pathlib.Path(path)

    def registrar(self, sesion: SesionTerapeutica) -> None:
        datos = json.loads(sesion.exportar_json())
        with self.path.open("a") as f:
            f.write(json.dumps(datos, ensure_ascii=False) + "\n")

    def replay(self) -> list[dict]:
        """Reconstruye toda la historia de intervenciones."""
        eventos = []
        with self.path.open() as f:
            for linea in f:
                eventos.append(json.loads(linea))
        return eventos
```

---

### MEJORA 7 (PROPUESTA ARQUITECTÓNICA) — Canal de Lindblad Generalizado

**Inspiración cuántica:** el canal de desfase actual (Kraus con solo 2 operadores)
es una aproximación. La ecuación maestra de Lindblad describe la decoherencia
más general, incluyendo relajación (T1) y desfase puro (T2), los dos mecanismos
físicos reales.

**Inspiración junguiana:** la represión tiene al menos dos mecanismos distintos:
- **T1 (relajación):** olvido activo — el contenido se "disipa" hacia el inconsciente.
- **T2 (desfase puro):** interferencia — el contenido permanece pero pierde coherencia
  de fase, imposibilitando el insight sin destruir el contenido.

**Implementación propuesta:**
```python
# En experiments.py — extensión de ParConDecoherencia
def aplicar_represion_lindblad(self, gamma1: float, gamma2: float, dt: float = 1.0) -> None:
    """
    Canal de Lindblad con relajación (γ₁) y desfase puro (γ₂).

    Operadores de salto:
        L1 = sqrt(γ₁) * σ₋ ⊗ I    (relajación: |1>→|0>, olvido activo)
        L2 = sqrt(γ₂/2) * σz ⊗ I  (desfase puro: pérdida de coherencia de fase)
    """
    I2 = np.eye(2)
    sig_minus = np.array([[0,1],[0,0]])
    sig_z     = np.array([[1,0],[0,-1]])
    L1 = np.sqrt(gamma1) * np.kron(sig_minus, I2)
    L2 = np.sqrt(gamma2 / 2) * np.kron(sig_z, I2)
    # Ecuación maestra: dρ/dt = Σ_k (Lk ρ Lk† - ½{Lk†Lk, ρ})
    drho = sum(
        L @ self.rho @ L.conj().T
        - 0.5 * (L.conj().T @ L @ self.rho + self.rho @ L.conj().T @ L)
        for L in [L1, L2]
    )
    self.rho += drho * dt
    # Re-normalizar para corregir errores numéricos de Euler
    self.rho /= np.trace(self.rho)
```

---

### MEJORA 8 (PROPUESTA ARQUITECTÓNICA) — Diagnóstico Arquetipal Bayesiano

**Inspiración cuántica:** la inferencia bayesiana es la contraparte clásica de la
tomografía de estado cuántico (QST). Dadas N mediciones binarias (0=Ánima, 1=Ánimus),
inferir la distribución posterior sobre α.

**Inspiración junguiana:** el clínico junguiano infiere el "estado arquetípico" de
un paciente a partir de observaciones conductuales (sueños, asociaciones libres,
proyecciones). Este módulo lo formaliza matemáticamente.

**Implementación propuesta:**
```python
# diagnostico.py — nuevo módulo
import numpy as np
from scipy.stats import beta as beta_dist

def inferir_alpha_bayesiano(observaciones: list[int],
                             prior_a: float = 1.0,
                             prior_b: float = 1.0) -> dict:
    """
    Inferencia bayesiana del parámetro p = α² (probabilidad de Ánima).

    Modelo: observaciones ~ Bernoulli(p), prior: p ~ Beta(a, b).
    Posterior analítica: p | datos ~ Beta(a + n_anima, b + n_animus).

    Args:
        observaciones: lista de 0s (Ánima) y 1s (Ánimus) medidos.
        prior_a, prior_b: parámetros del prior Beta (default: uniforme).

    Returns:
        dict con media, moda, IC 95% y alpha_MAP = sqrt(p_MAP).
    """
    n_anima  = sum(1 for x in observaciones if x == 0)
    n_animus = len(observaciones) - n_anima
    a_post   = prior_a + n_anima
    b_post   = prior_b + n_animus
    dist     = beta_dist(a_post, b_post)

    p_media = dist.mean()
    p_moda  = (a_post - 1) / (a_post + b_post - 2) if a_post > 1 else 0.0
    ic_95   = dist.interval(0.95)

    return {
        "p_media":    round(p_media, 4),
        "p_moda":     round(p_moda, 4),
        "alpha_MAP":  round(np.sqrt(p_moda), 4),
        "IC_95_p":    (round(ic_95[0], 4), round(ic_95[1], 4)),
        "IC_95_alpha":(round(np.sqrt(ic_95[0]), 4), round(np.sqrt(ic_95[1]), 4)),
        "n_obs":      len(observaciones),
    }

# Uso:
# obs = [arq.medir() for _ in range(100)]
# print(inferir_alpha_bayesiano(obs))
```

---

## PARTE III — ARQUITECTURA RECOMENDADA v3.0

```
laboratorio_cuantico/
├── core/
│   ├── experiments.py      ← Qubit, Par, entrelazamiento (CORREGIDO)
│   ├── archetypes.py       ← 5 arquetipos + RegistroCuantico (CORREGIDO)
│   └── interventions.py    ← Puertas + SesionTerapeutica (CORREGIDO)
│
├── ml/
│   ├── collect_data.py     ← Datasets A, B, C (CORREGIDO)
│   ├── train_regression.py ← Lineal, Poli, MLP, GridSearch (CORREGIDO)
│   └── analysis.py         ← Gráficas + radar + heatmap (CORREGIDO)
│
├── nuevos/                  ← PROPUESTOS
│   ├── diagnostico.py      ← Inferencia bayesiana del α
│   ├── events.py           ← Event Sourcing / DiarioIndividuacion
│   └── lindblad.py         ← Canal generalizado T1/T2
│
├── streamlit_app.py        ← Dashboard (ya corregido)
├── main.py                 ← CLI (CORREGIDO + ampliado)
└── requirements.txt        ← Actualizado
```

---

## PARTE IV — TABLA RESUMEN DE BUGS

| ID | Severidad | Archivo | Descripción | Estado |
|----|-----------|---------|-------------|--------|
| B-01 | 🔴 CRÍTICO | experiments.py | RNG global contamina todo el proceso | ✅ CORREGIDO |
| B-02 | 🔴 CRÍTICO | streamlit_app.py | Doble medir_base_X() — correlación siempre ≈0.5 | ✅ CORREGIDO |
| B-03 | 🔴 CRÍTICO | experiments.py | aplicar_rotacion genera objeto sin `_rng` | ✅ CORREGIDO |
| B-04 | 🔴 CRÍTICO | collect_data.py | seed global al importar el módulo | ✅ CORREGIDO |
| M-01 | 🟡 MENOR | archetypes.py | AnimaAnimus(1.0, 1.0) implícito | ✅ CORREGIDO |
| M-02 | 🟡 MENOR | archetypes.py | indice_individuacion formula inestable | ✅ CORREGIDO |
| M-03 | 🟡 MENOR | train_regression.py | Sin guard de existencia de CSV | ✅ CORREGIDO |
| M-04 | 🟡 MENOR | analysis.py | Sin guard de existencia de .pkl | ✅ CORREGIDO |
| M-05 | 🟡 MENOR | interventions.py | Sin captura de TypeError en kwargs | ✅ CORREGIDO |
| M-06 | 🟡 MENOR | main.py | Imports tardíos dentro de funciones | ✅ CORREGIDO |

---

> *"El bug es la Sombra del código: opera en silencio, contamina el sistema
> y sólo revela su presencia cuando el observador mide en el ángulo correcto."*
