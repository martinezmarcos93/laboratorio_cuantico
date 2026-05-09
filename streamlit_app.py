"""
streamlit_app.py — Dashboard interactivo del Laboratorio Cuántico-Junguiano.

Ejecutar con:
    streamlit run streamlit_app.py

Secciones:
  1. Superposición del Arquetipo  — slider de alpha, predicción en tiempo real
  2. Sincronicidad bajo represión — slider de gamma, correlación estimada
  3. Registro cuántico            — psique completa con 5 componentes
  4. Sesión terapéutica           — pipeline de puertas cuánticas interactivo
  5. Comparación de modelos       — tabla y gráfica de todos los modelos ML

BUGS CORREGIDOS respecto a la versión original:
  1. [CRÍTICO] simular_sincronicidad: comparaba par.medir_base_X()[0] con
     par.medir_base_X()[1], que son dos llamadas DISTINTAS (dos mediciones
     independientes), no los dos qubits del mismo par. Corregido con
     x1, x2 = par.medir_base_X() y comparación x1 == x2.
  2. [CRÍTICO] cargar_modelos: usaba rutas de archivo incorrectas que no
     coincidían con las guardadas en train_regression.py.
     Corregido: regresion_arquetipo_lineal.pkl, regresion_arquetipo_poli.pkl,
     regresion_sincronicidad.pkl.
  3. [CRÍTICO] archetypes.py e interventions.py importados pero inexistentes.
     Creados como módulos completos.
  4. [MENOR] train_extended.py referenciado en sección 5 pero inexistente.
     Reemplazado por lógica inline que usa train_regression.py existente.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Asegurar que el directorio del proyecto esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments   import Arquetipo, ParConDecoherencia
from archetypes    import Persona, Sombra, SiMismo, RegistroCuantico
from interventions import (
    apertura_consciente, integracion_parcial, amplificacion,
    proyeccion, SesionTerapeutica,
)

# ─────────────────────────────────────────────
# Configuración global
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Laboratorio Cuántico-Junguiano",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-box {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .big-number {
        font-size: 2.2rem;
        font-weight: 700;
        color: #cba6f7;
    }
    .label {
        font-size: 0.85rem;
        color: #a6adc8;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def cargar_modelos():
    """
    Carga los modelos entrenados si existen; retorna None si no.

    BUG FIX: las rutas originales eran incorrectas:
      ✗ modelos/arquetipo_lineal.pkl
      ✗ modelos/arquetipo_polinomial_2.pkl
      ✗ modelos/sincronicidad_lineal.pkl
    Corregidas a las rutas reales usadas en train_regression.py:
      ✓ modelos/regresion_arquetipo_lineal.pkl
      ✓ modelos/regresion_arquetipo_poli.pkl
      ✓ modelos/regresion_sincronicidad.pkl
    """
    try:
        import joblib
        lin  = joblib.load("modelos/regresion_arquetipo_lineal.pkl")
        poli = joblib.load("modelos/regresion_arquetipo_poli.pkl")
        sinc = joblib.load("modelos/regresion_sincronicidad.pkl")
        return lin, poli, sinc
    except Exception:
        return None, None, None


def simular_arquetipo(alpha: float, n_shots: int = 1000) -> float:
    """Estima P(Ánima) para un alpha dado."""
    beta = np.sqrt(max(0.0, 1 - alpha ** 2))
    arq  = Arquetipo(alpha, beta)
    return sum(1 - arq.medir() for _ in range(n_shots)) / n_shots


def simular_sincronicidad(gamma: float, n_trials: int = 500) -> float:
    """
    Estima la correlación en base X para un gamma dado.

    BUG FIX: el código original hacía:
        sum(1 for _ in range(n_trials)
            if par.medir_base_X()[0] == par.medir_base_X()[1])
    Esto llama a medir_base_X() DOS VECES por iteración, comparando
    el qubit 1 de una medición con el qubit 2 de una medición DIFERENTE.
    El resultado es una correlación aleatoria ≈ 0.5 independiente de gamma.

    Corrección: una sola llamada a medir_base_X() retorna ambos resultados.
    """
    par = ParConDecoherencia()
    par.aplicar_represion(gamma)
    iguales = 0
    for _ in range(n_trials):
        x1, x2 = par.medir_base_X()   # ← una sola llamada; ambos qubits del mismo par
        if x1 == x2:
            iguales += 1
    return iguales / n_trials


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("🧬 Lab Cuántico-Junguiano")
    st.markdown("---")
    seccion = st.radio(
        "Sección",
        [
            "🔮 Superposición del Arquetipo",
            "🌀 Sincronicidad bajo represión",
            "🧠 Registro cuántico",
            "💊 Sesión terapéutica",
            "📊 Comparación de modelos ML",
        ],
    )
    st.markdown("---")
    st.caption("Laboratorio para explorar física cuántica y psicología analítica.")


# ═══════════════════════════════════════════════════════
# SECCIÓN 1 — Superposición del Arquetipo
# ═══════════════════════════════════════════════════════

if seccion == "🔮 Superposición del Arquetipo":
    st.header("🔮 Superposición del Arquetipo")
    st.markdown(
        "Variar `α` cambia la amplitud del polo **Ánima** (`|0⟩`). "
        "La probabilidad teórica sigue **P(Ánima) = α²**."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        alpha   = st.slider("α (amplitud Ánima)", 0.0, 1.0, 0.7, 0.01)
        n_shots = st.select_slider("Mediciones (n_shots)", [100, 500, 1000, 5000], value=1000)
        simular = st.button("⚡ Simular medición", use_container_width=True)

        beta      = np.sqrt(max(0.0, 1 - alpha ** 2))
        p_teorica = alpha ** 2
        p_animus  = 1 - p_teorica

        st.metric("P(Ánima) teórica",  f"{p_teorica:.4f}")
        st.metric("P(Ánimus) teórica", f"{p_animus:.4f}")
        st.metric("β (amplitud Ánimus)", f"{beta:.4f}")

        if simular:
            with st.spinner("Simulando..."):
                p_obs = simular_arquetipo(alpha, n_shots)
            st.metric("P(Ánima) observada", f"{p_obs:.4f}", delta=f"{p_obs - p_teorica:+.4f}")

    with col2:
        alpha_range = np.linspace(0, 1, 200)
        fig, ax     = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        ax.plot(alpha_range, alpha_range ** 2, color="#cba6f7", lw=2.5, label="Teórico: P(0) = α²")
        ax.axvline(alpha,     color="#f38ba8", ls="--", lw=1.5, label=f"α = {alpha:.2f}")
        ax.axhline(p_teorica, color="#a6e3a1", ls=":",  lw=1.2, label=f"P(Ánima) = {p_teorica:.4f}")
        ax.scatter([alpha], [p_teorica], color="#f38ba8", s=80, zorder=5)

        ax.set_xlabel("α", color="#cdd6f4")
        ax.set_ylabel("P(Ánima = |0⟩)", color="#cdd6f4")
        ax.set_title("Curva de probabilidad del arquetipo", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        ax.spines[:].set_color("#45475a")
        ax.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4", edgecolor="#45475a")
        st.pyplot(fig)

    st.markdown("### Tabla teórica")
    alphas = np.arange(0, 1.05, 0.1)
    df_tab = pd.DataFrame({
        "α":         alphas.round(2),
        "β":         np.sqrt(np.maximum(0, 1 - alphas**2)).round(4),
        "P(Ánima)":  (alphas**2).round(4),
        "P(Ánimus)": (1 - alphas**2).round(4),
    })
    st.dataframe(df_tab, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# SECCIÓN 2 — Sincronicidad bajo represión
# ═══════════════════════════════════════════════════════

elif seccion == "🌀 Sincronicidad bajo represión":
    st.header("🌀 Sincronicidad bajo represión (decoherencia)")
    st.markdown(
        "El parámetro `γ` controla la intensidad del canal de desfase sobre el primer qubit. "
        "La correlación teórica en base X sigue **correlación = 1 − γ**."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        gamma    = st.slider("γ (nivel de represión)", 0.0, 1.0, 0.3, 0.01)
        n_trials = st.select_slider("Trials por estimación", [200, 500, 1000], value=500)
        simular2 = st.button("⚡ Estimar correlación", use_container_width=True)

        c_teorica = 1 - gamma
        st.metric("Correlación teórica", f"{c_teorica:.4f}")

        if simular2:
            with st.spinner("Simulando entrelazamiento..."):
                c_obs = simular_sincronicidad(gamma, n_trials)
            st.metric("Correlación observada", f"{c_obs:.4f}", delta=f"{c_obs - c_teorica:+.4f}")

        if gamma < 0.2:
            st.success("🌿 Entrelazamiento casi intacto (sincronicidad alta)")
        elif gamma < 0.6:
            st.warning("⚠️ Decoherencia parcial (represión moderada)")
        else:
            st.error("🔴 Decoherencia severa (sincronicidad baja)")

    with col2:
        gammas_range = np.linspace(0, 1, 200)
        fig2, ax2    = plt.subplots(figsize=(7, 4))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#0e1117")

        ax2.plot(gammas_range, 1 - gammas_range, color="#89b4fa", lw=2.5, label="Teórico: 1 − γ")
        ax2.axvline(gamma,     color="#f38ba8", ls="--", lw=1.5, label=f"γ = {gamma:.2f}")
        ax2.axhline(c_teorica, color="#a6e3a1", ls=":",  lw=1.2, label=f"Corr = {c_teorica:.4f}")
        ax2.scatter([gamma], [c_teorica], color="#f38ba8", s=80, zorder=5)

        ax2.set_xlabel("γ (represión)", color="#cdd6f4")
        ax2.set_ylabel("Correlación en base X", color="#cdd6f4")
        ax2.set_title("Sincronicidad bajo decoherencia", color="#cdd6f4")
        ax2.tick_params(colors="#cdd6f4")
        ax2.spines[:].set_color("#45475a")
        ax2.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4", edgecolor="#45475a")
        st.pyplot(fig2)


# ═══════════════════════════════════════════════════════
# SECCIÓN 3 — Registro cuántico
# ═══════════════════════════════════════════════════════

elif seccion == "🧠 Registro cuántico":
    st.header("🧠 Registro cuántico — Psique completa")
    st.markdown(
        "El **Registro Cuántico** modela los 5 componentes junguianos clásicos "
        "como qubits con amplitudes iniciales características."
    )

    seed_reg = st.number_input("Semilla aleatoria", value=42, min_value=0, max_value=9999)

    if "registro" not in st.session_state or st.button("🔄 Reiniciar registro"):
        st.session_state.registro    = RegistroCuantico(seed=int(seed_reg))
        st.session_state.mediciones  = []

    reg = st.session_state.registro

    coherencia        = reg.coherencia_global()
    componentes_data  = []
    for nombre, qubit in zip(reg.COMPONENTES, reg.qubits):
        componentes_data.append({
            "Componente":   nombre,
            "P(Ánima)":     round(abs(qubit.alpha)**2, 4),
            "P(Ánimus)":    round(abs(qubit.beta)**2,  4),
            "Entropía (H)": round(coherencia[nombre],  4),
        })
    st.dataframe(pd.DataFrame(componentes_data), use_container_width=True, hide_index=True)

    tension = reg.tension_yo_sombra()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Tensión Yo↔Sombra",    f"{tension:.4f}")
    col_b.metric("H(SiMismo)",           f"{coherencia['SiMismo']:.4f} bits")
    col_c.metric("Mediciones realizadas", len(st.session_state.mediciones))

    if st.button("🎲 Medir todos los componentes"):
        m = reg.medir_todo()
        st.session_state.mediciones.append(m)
        st.success(f"Resultado: {m}")

    if st.session_state.mediciones:
        st.markdown("### Historial de mediciones")
        st.dataframe(pd.DataFrame(st.session_state.mediciones), use_container_width=True)


# ═══════════════════════════════════════════════════════
# SECCIÓN 4 — Sesión terapéutica
# ═══════════════════════════════════════════════════════

elif seccion == "💊 Sesión terapéutica":
    st.header("💊 Sesión terapéutica — Pipeline de intervenciones")
    st.markdown(
        "Construí una secuencia de **puertas cuánticas** aplicadas a un arquetipo inicial. "
        "Cada intervención transforma el estado y se registra en el historial."
    )

    col_i, col_s = st.columns([1, 2])

    with col_i:
        alpha_ini = st.slider("α inicial", 0.0, 1.0, 0.95, 0.01, key="terapia_alpha")
        beta_ini  = np.sqrt(max(0.0, 1 - alpha_ini**2))
        st.caption(f"β inicial = {beta_ini:.4f}")

        if "sesion" not in st.session_state or st.button("🔄 Reiniciar sesión"):
            arq_ini = Arquetipo(alpha_ini, beta_ini)
            st.session_state.sesion = SesionTerapeutica(arq_ini)

        sesion = st.session_state.sesion

        st.markdown("#### Añadir intervención")
        intervencion = st.selectbox(
            "Tipo de intervención",
            [
                "apertura_consciente",
                "integracion_parcial",
                "amplificacion",
                "proyeccion_anima",
                "proyeccion_animus",
            ]
        )

        theta_val = np.pi / 4
        polo_val  = 0
        if intervencion in ("integracion_parcial", "amplificacion"):
            theta_deg = st.slider("θ (grados)", 0, 360, 45, 5)
            theta_val = np.deg2rad(theta_deg)
        if intervencion == "amplificacion":
            polo_val = st.radio(
                "Polo a amplificar", [0, 1],
                format_func=lambda x: "Ánima (|0>)" if x == 0 else "Ánimus (|1>)"
            )

        if st.button("➕ Aplicar intervención", use_container_width=True):
            kwargs = {}
            if intervencion in ("integracion_parcial", "amplificacion"):
                kwargs["theta"] = theta_val
            if intervencion == "amplificacion":
                kwargs["polo"] = polo_val
            try:
                sesion.aplicar(intervencion, **kwargs)
                st.success(f"Aplicado: {intervencion}")
            except ValueError as e:
                st.error(str(e))

    with col_s:
        if sesion.historial:
            hist_df = pd.DataFrame(sesion.historial)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

            fig3, ax3 = plt.subplots(figsize=(7, 3.5))
            fig3.patch.set_facecolor("#0e1117")
            ax3.set_facecolor("#0e1117")

            pasos    = range(len(sesion.historial))
            p_animas = [h["P_anima"]  for h in sesion.historial]
            p_animus = [h["P_animus"] for h in sesion.historial]

            ax3.plot(pasos, p_animas, "o-", color="#cba6f7", lw=2, label="P(Ánima)")
            ax3.plot(pasos, p_animus, "s-", color="#f38ba8", lw=2, label="P(Ánimus)")
            ax3.axhline(0.5, color="#45475a", ls=":", lw=1)
            ax3.set_xticks(list(pasos))
            ax3.set_xticklabels(
                [h["intervencion"][:12] for h in sesion.historial],
                rotation=30, ha="right", color="#cdd6f4", fontsize=8
            )
            ax3.set_ylim(-0.05, 1.05)
            ax3.set_ylabel("Probabilidad", color="#cdd6f4")
            ax3.set_title("Evolución del arquetipo durante la sesión", color="#cdd6f4")
            ax3.tick_params(colors="#cdd6f4")
            ax3.spines[:].set_color("#45475a")
            ax3.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4", edgecolor="#45475a")
            st.pyplot(fig3)


# ═══════════════════════════════════════════════════════
# SECCIÓN 5 — Comparación de modelos ML
# ═══════════════════════════════════════════════════════

elif seccion == "📊 Comparación de modelos ML":
    st.header("📊 Comparación de modelos ML")
    st.markdown(
        "Compara el rendimiento de los modelos Lineal y Polinomial-2 "
        "sobre ambos datasets. Generá los datos desde el menú o `main.py`."
    )

    # BUG FIX: train_extended.py no existe. Reemplazado por train_regression.py existente.
    entrenar = st.button("🚀 Generar datos y entrenar modelos ahora")
    if entrenar:
        with st.spinner("Generando datasets..."):
            from collect_data import generar_dataset_arquetipo, generar_dataset_sincronicidad
            os.makedirs("datasets", exist_ok=True)
            generar_dataset_arquetipo()
            generar_dataset_sincronicidad()
        with st.spinner("Entrenando modelos..."):
            from train_regression import entrenar_modelos_arquetipo, entrenar_modelo_sincronicidad
            os.makedirs("modelos", exist_ok=True)
            lin_a, poli_a = entrenar_modelos_arquetipo()
            lin_s         = entrenar_modelo_sincronicidad()
        st.success("✅ Modelos entrenados y guardados.")

    lin, poli, sinc = cargar_modelos()

    if lin is not None:
        tab1, tab2 = st.tabs(["Arquetipo (P = α²)", "Sincronicidad (corr = 1−γ)"])

        with tab1:
            try:
                df_a       = pd.read_csv("datasets/arquetipo_prob.csv")
                x_range    = np.linspace(0, 1, 200).reshape(-1, 1)
                fig4, ax4  = plt.subplots(figsize=(8, 4.5))
                fig4.patch.set_facecolor("#0e1117")
                ax4.set_facecolor("#0e1117")

                ax4.scatter(df_a["alpha"], df_a["prob_anima"],
                            alpha=0.5, color="#89dceb", s=20, label="Datos simulados", zorder=3)
                ax4.plot(x_range, x_range**2,              "w:", lw=1.5, label="Teórico: α²")
                ax4.plot(x_range, lin.predict(x_range),  color="#f38ba8", lw=1.8, label="Lineal")
                ax4.plot(x_range, poli.predict(x_range), color="#a6e3a1", lw=1.8, label="Polinomial-2")

                ax4.set_xlabel("α", color="#cdd6f4")
                ax4.set_ylabel("P(Ánima)", color="#cdd6f4")
                ax4.set_title("Superposición arquetípica", color="#cdd6f4")
                ax4.tick_params(colors="#cdd6f4")
                ax4.spines[:].set_color("#45475a")
                ax4.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4", edgecolor="#45475a")
                st.pyplot(fig4)
            except FileNotFoundError:
                st.warning("Dataset no encontrado. Entrenálo primero.")

        with tab2:
            try:
                df_s       = pd.read_csv("datasets/sincronicidad_corr.csv")
                g_range    = np.linspace(0, 1, 200).reshape(-1, 1)
                fig5, ax5  = plt.subplots(figsize=(8, 4.5))
                fig5.patch.set_facecolor("#0e1117")
                ax5.set_facecolor("#0e1117")

                ax5.scatter(df_s["gamma"], df_s["correlacion"],
                            alpha=0.5, color="#89dceb", s=20, label="Datos simulados", zorder=3)
                ax5.plot(g_range, 1 - g_range,            "w:", lw=1.5, label="Teórico: 1−γ")
                ax5.plot(g_range, sinc.predict(g_range), color="#f38ba8", lw=1.8, label="Lineal")

                ax5.set_xlabel("γ", color="#cdd6f4")
                ax5.set_ylabel("Correlación en base X", color="#cdd6f4")
                ax5.set_title("Sincronicidad bajo decoherencia", color="#cdd6f4")
                ax5.tick_params(colors="#cdd6f4")
                ax5.spines[:].set_color("#45475a")
                ax5.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4", edgecolor="#45475a")
                st.pyplot(fig5)
            except FileNotFoundError:
                st.warning("Dataset no encontrado. Entrenálo primero.")
    else:
        st.info("Hacé clic en '🚀 Generar datos y entrenar modelos ahora' para comenzar.")
