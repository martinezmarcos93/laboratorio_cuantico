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
            "🕸️ Grafo de Sincronicidad",
            "🔬 Diagnóstico Bayesiano",
            "📋 Informe Clínico (IA)",
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
        plt.close(fig)

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
        plt.close(fig2)


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

            pasos      = list(range(len(sesion.historial)))
            p_animas   = [h["P_anima"]   for h in sesion.historial]
            p_animus   = [h["P_animus"]  for h in sesion.historial]
            entropias  = [h["entropia"]  for h in sesion.historial]
            etiquetas  = [h["intervencion"][:12] for h in sesion.historial]

            fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(7, 6),
                                               gridspec_kw={"hspace": 0.45})
            for ax in (ax3a, ax3b):
                ax.set_facecolor("#0e1117")
            fig3.patch.set_facecolor("#0e1117")

            # Subgráfica 1: P(Ánima) y P(Ánimus)
            ax3a.plot(pasos, p_animas, "o-", color="#cba6f7", lw=2, label="P(Ánima)")
            ax3a.plot(pasos, p_animus, "s-", color="#f38ba8", lw=2, label="P(Ánimus)")
            ax3a.axhline(0.5, color="#45475a", ls=":", lw=1)
            ax3a.set_xticks(pasos)
            ax3a.set_xticklabels(etiquetas, rotation=30, ha="right",
                                 color="#cdd6f4", fontsize=8)
            ax3a.set_ylim(-0.05, 1.05)
            ax3a.set_ylabel("Probabilidad", color="#cdd6f4")
            ax3a.set_title("Evolución del arquetipo", color="#cdd6f4")
            ax3a.tick_params(colors="#cdd6f4")
            ax3a.spines[:].set_color("#45475a")
            ax3a.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4", edgecolor="#45475a",
                        fontsize=8)

            # Subgráfica 2: Entropía = índice de ambigüedad / individuación
            ax3b.plot(pasos, entropias, "^-", color="#a6e3a1", lw=2, ms=7,
                      label="Entropía (ambigüedad psíquica)")
            ax3b.axhline(1.0, color="#45475a", ls="--", lw=1, label="Máximo (1 bit)")
            ax3b.fill_between(pasos, entropias, alpha=0.15, color="#a6e3a1")
            ax3b.set_xticks(pasos)
            ax3b.set_xticklabels(etiquetas, rotation=30, ha="right",
                                 color="#cdd6f4", fontsize=8)
            ax3b.set_ylim(-0.05, 1.15)
            ax3b.set_ylabel("H Shannon (bits)", color="#cdd6f4")
            ax3b.set_title("Índice de individuación (entropía)", color="#cdd6f4")
            ax3b.tick_params(colors="#cdd6f4")
            ax3b.spines[:].set_color("#45475a")
            ax3b.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4", edgecolor="#45475a",
                        fontsize=8)

            st.pyplot(fig3)
            plt.close(fig3)


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

                try:
                    degree     = poli.named_steps["polynomialfeatures"].degree
                    label_poli = f"Polinomial (grado {degree})"
                except Exception:
                    label_poli = "Polinomial"

                ax4.scatter(df_a["alpha"], df_a["prob_anima"],
                            alpha=0.5, color="#89dceb", s=20, label="Datos simulados", zorder=3)
                ax4.plot(x_range, x_range**2,              "w:", lw=1.5, label="Teórico: α²")
                ax4.plot(x_range, lin.predict(x_range),  color="#f38ba8", lw=1.8, label="Lineal")
                ax4.plot(x_range, poli.predict(x_range), color="#a6e3a1", lw=1.8, label=label_poli)

                ax4.set_xlabel("α", color="#cdd6f4")
                ax4.set_ylabel("P(Ánima)", color="#cdd6f4")
                ax4.set_title("Superposición arquetípica", color="#cdd6f4")
                ax4.tick_params(colors="#cdd6f4")
                ax4.spines[:].set_color("#45475a")
                ax4.legend(facecolor="#1e1e2e", labelcolor="#cdd6f4", edgecolor="#45475a")
                st.pyplot(fig4)
                plt.close(fig4)
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
                plt.close(fig5)
            except FileNotFoundError:
                st.warning("Dataset no encontrado. Entrenálo primero.")
    else:
        st.info("Hacé clic en '🚀 Generar datos y entrenar modelos ahora' para comenzar.")


# ═══════════════════════════════════════════════════════
# SECCIÓN 6 — Grafo de Sincronicidad
# ═══════════════════════════════════════════════════════

elif seccion == "🕸️ Grafo de Sincronicidad":
    st.header("🕸️ Grafo de Sincronicidad — Red arquetípica")
    st.markdown(
        "La **matriz de fidelidades** entre los 5 componentes de la psique visualiza "
        "la resonancia arquetípica: F alto = componentes muy similares (posible proyección); "
        "F bajo = componentes ortogonales (tensión creativa)."
    )

    seed_g = st.number_input("Semilla", value=42, min_value=0, max_value=9999, key="seed_grafo")
    reg_g  = RegistroCuantico(seed=int(seed_g))
    mat    = reg_g.grafo_sincronicidad()
    nombres = reg_g.COMPONENTES

    col_h, col_n = st.columns(2)

    with col_h:
        st.markdown("#### Heatmap de fidelidades")
        fig_h, ax_h = plt.subplots(figsize=(5, 4))
        fig_h.patch.set_facecolor("#0e1117")
        ax_h.set_facecolor("#0e1117")
        im = ax_h.imshow(mat, cmap="viridis", vmin=0, vmax=1)
        fig_h.colorbar(im, ax=ax_h, label="F = |⟨ψᵢ|ψⱼ⟩|²")
        ax_h.set_xticks(range(5)); ax_h.set_xticklabels(nombres, rotation=45,
                                                          ha="right", color="#cdd6f4", fontsize=8)
        ax_h.set_yticks(range(5)); ax_h.set_yticklabels(nombres, color="#cdd6f4", fontsize=8)
        ax_h.set_title("Resonancia inter-arquetípica", color="#cdd6f4")
        ax_h.tick_params(colors="#cdd6f4")
        ax_h.spines[:].set_color("#45475a")
        st.pyplot(fig_h)
        plt.close(fig_h)

    with col_n:
        st.markdown("#### Grafo de resonancia")
        try:
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(nombres)
            umbral = st.slider("Umbral mínimo de fidelidad", 0.0, 1.0, 0.5, 0.05)
            for i in range(5):
                for j in range(i + 1, 5):
                    if mat[i, j] >= umbral:
                        G.add_edge(nombres[i], nombres[j], weight=float(mat[i, j]))

            fig_n, ax_n = plt.subplots(figsize=(5, 4))
            fig_n.patch.set_facecolor("#0e1117")
            ax_n.set_facecolor("#0e1117")
            pos    = nx.circular_layout(G)
            edges  = list(G.edges(data="weight", default=0))
            widths = [w * 6 for _, _, w in edges] if edges else []
            nx.draw_networkx_nodes(G, pos, ax=ax_n, node_color="#6c63ff",
                                   node_size=700, alpha=0.9)
            nx.draw_networkx_labels(G, pos, ax=ax_n, font_color="#cdd6f4",
                                    font_size=8)
            if edges:
                nx.draw_networkx_edges(G, pos, ax=ax_n, width=widths,
                                       edge_color="#a6e3a1", alpha=0.7)
            ax_n.set_title(f"Aristas con F ≥ {umbral:.2f}", color="#cdd6f4")
            ax_n.axis("off")
            st.pyplot(fig_n)
            plt.close(fig_n)
        except ImportError:
            st.warning("networkx no instalado. Ejecutá: `pip install networkx`")

    st.markdown("### Métricas del registro")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Índice de individuación", f"{reg_g.indice_individuacion():.4f}")
    col_m2.metric("Tensión Yo↔Sombra",        f"{reg_g.tension_yo_sombra():.4f}")
    col_m3.metric("Narrativa",                 reg_g.narrativa_estado()[:60] + "…")
    st.caption(reg_g.narrativa_estado())


# ═══════════════════════════════════════════════════════
# SECCIÓN 7 — Diagnóstico Bayesiano
# ═══════════════════════════════════════════════════════

elif seccion == "🔬 Diagnóstico Bayesiano":
    st.header("🔬 Diagnóstico Arquetipal Bayesiano")
    st.markdown(
        "Infiere la distribución posterior sobre **α** de cada componente psíquico "
        "a partir de N mediciones. Modelo: _obs ~ Bernoulli(p)_, prior: _p ~ Beta(1,1)_ "
        "(uniforme). Posterior analítica: _p | datos ~ Beta(1 + n_ánima, 1 + n_ánimus)_."
    )

    try:
        from diagnostico import inferir_alpha, diagnosticar_registro, graficar_posterior

        seed_d  = st.number_input("Semilla", value=42, min_value=0, max_value=9999, key="seed_diag")
        n_obs_d = st.slider("Observaciones por componente", 20, 500, 200, 10)
        correr  = st.button("🔬 Ejecutar diagnóstico")

        if correr or "diag_resultados" in st.session_state:
            if correr:
                reg_d = RegistroCuantico(seed=int(seed_d))
                st.session_state.diag_resultados  = diagnosticar_registro(reg_d, n_obs=n_obs_d)
                st.session_state.diag_reg         = reg_d

            res_d = st.session_state.diag_resultados
            reg_d = st.session_state.diag_reg

            # Tabla resumen
            filas = []
            for nom, r in res_d.items():
                filas.append({
                    "Componente":  nom,
                    "α_MAP":       r["alpha_MAP"],
                    "p_media":     r["p_media"],
                    "IC 95% α":    f"[{r['IC_95_alpha'][0]}, {r['IC_95_alpha'][1]}]",
                    "n_obs":       r["n_obs"],
                })
            st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

            # Comparar con α teórico
            st.markdown("### α real vs α estimado")
            alphas_reales = [float(abs(q.alpha)) for q in reg_d.qubits]
            alphas_mape   = [r["alpha_MAP"] for r in res_d.values()]
            comp_df = pd.DataFrame({
                "Componente": reg_d.COMPONENTES,
                "α real":     [round(a, 4) for a in alphas_reales],
                "α MAP":      alphas_mape,
                "Error":      [round(abs(a - b), 4) for a, b in
                               zip(alphas_reales, alphas_mape)],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Posterior de un componente seleccionado
            st.markdown("### Distribución posterior detallada")
            comp_sel = st.selectbox("Componente", reg_d.COMPONENTES, key="comp_post")
            idx_sel  = reg_d.COMPONENTES.index(comp_sel)
            qubit_sel = reg_d.qubits[idx_sel]
            obs_sel   = [qubit_sel.medir() for _ in range(n_obs_d)]

            fig_p, ax_p = plt.subplots(figsize=(7, 3.5))
            fig_p.patch.set_facecolor("#0e1117")
            ax_p.set_facecolor("#0e1117")
            ax_p.tick_params(colors="#cdd6f4")
            ax_p.spines[:].set_color("#45475a")
            graficar_posterior(obs_sel, titulo=f"Posterior — {comp_sel}", ax=ax_p)
            ax_p.set_xlabel(ax_p.get_xlabel(), color="#cdd6f4")
            ax_p.set_ylabel(ax_p.get_ylabel(), color="#cdd6f4")
            ax_p.set_title(ax_p.get_title(), color="#cdd6f4")
            st.pyplot(fig_p)
            plt.close(fig_p)

    except ImportError as e:
        st.error(f"No se pudo cargar diagnostico.py: {e}\n"
                 "Verificá que scipy esté instalado: `pip install scipy`")


# ═══════════════════════════════════════════════════════
# SECCIÓN 8 — Informe Clínico (IA)
# ═══════════════════════════════════════════════════════

elif seccion == "📋 Informe Clínico (IA)":
    st.header("📋 Informe Clínico — Interpretación por Claude")
    st.markdown(
        "Genera un **informe clínico narrativo** en lenguaje natural a partir del estado "
        "cuántico de un RegistroCuantico y una sesión terapéutica opcional. "
        "Requiere `ANTHROPIC_API_KEY` configurada como variable de entorno."
    )

    with st.expander("⚙️ Configuración"):
        seed_i  = st.number_input("Semilla del registro", value=42, min_value=0,
                                   max_value=9999, key="seed_inf")
        alpha_i = st.slider("α inicial de la sesión (opcional)", 0.0, 1.0, 0.90, 0.01,
                             key="alpha_inf")
        usar_sesion = st.checkbox("Incluir sesión terapéutica en el informe", value=True)
        modelo_inf  = st.selectbox("Modelo de Claude",
                                    ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
                                    key="modelo_inf")

    generar = st.button("📋 Generar informe clínico", use_container_width=True)

    if generar:
        try:
            from informe_analitico import generar_informe_con_cache

            reg_i = RegistroCuantico(seed=int(seed_i))
            sesion_i = None
            if usar_sesion:
                beta_i   = float(np.sqrt(max(0.0, 1.0 - alpha_i**2)))
                arq_i    = Arquetipo(alpha_i, beta_i)
                sesion_i = SesionTerapeutica(arq_i)
                sesion_i.aplicar("apertura_consciente")
                sesion_i.aplicar("integracion_parcial", theta=np.pi / 4)

            with st.spinner("Consultando Claude API..."):
                informe = generar_informe_con_cache(reg_i, sesion_i, modelo=modelo_inf)

            st.success("✅ Informe generado")
            st.markdown("---")
            st.markdown(informe)
            st.markdown("---")
            st.download_button(
                "⬇️ Descargar informe (.txt)",
                data=informe,
                file_name="informe_clinico.txt",
                mime="text/plain",
            )

        except ImportError as e:
            st.error(f"Error de importación: {e}")
        except ValueError as e:
            st.error(str(e))
            st.info(
                "Para configurar la API key en Windows PowerShell:\n\n"
                "```powershell\n$env:ANTHROPIC_API_KEY = 'tu-clave-aqui'\n```\n\n"
                "Luego relanzá Streamlit desde la misma terminal."
            )
        except Exception as e:
            st.error(f"Error al generar el informe: {e}")
