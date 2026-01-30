import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib

# 1. CONFIGURACIÓN
st.set_page_config(page_title="Quantum Data Force | UPTC", page_icon="⚡", layout="wide")

st.title("⚡ Inteligencia Energética UPTC (2018 - 2025)")
st.markdown("**Análisis de Tendencias e IA** | Modelo v3")

# 2. CARGA DE DATOS
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    # Ruta al ZIP en la carpeta datos
    file_path = os.path.join(base_path, '../datos/consumos_uptc.zip')
    try:
        df = pd.read_csv(file_path, compression='zip')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Aseguramos que existan columnas numéricas para las sedes
        if 'sede' in df.columns:
            df['sede_n'] = df['sede'].astype('category').cat.codes
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

df = load_data()

if df is not None:
    # --- PANEL LATERAL ---
    st.sidebar.header("🕹️ Controles")
    sede_selec = st.sidebar.selectbox("Selecciona Sede:", df['sede'].unique())
    vista = st.sidebar.radio("Resolución histórica:", ["Mensual", "Diaria"])
    
    df_sede = df[df['sede'] == sede_selec].copy()
    
    # -----------------------------------------------------------------------------
    # 3. TENDENCIA 2018 - 2025
    # -----------------------------------------------------------------------------
    st.subheader(f"📈 Evolución Histórica: {sede_selec}")
    
    resample_rule = 'M' if vista == "Mensual" else 'D'
    # Agrupamos solo columnas numéricas para evitar errores
    df_hist = df_sede.set_index('timestamp').select_dtypes(include=['number']).resample(resample_rule).mean().reset_index()

    fig_hist = px.line(df_hist, x='timestamp', y='energia_total_kwh', 
                        title=f"Consumo Promedio ({vista})",
                        color_discrete_sequence=['#1ABC9C'])
    fig_hist.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_hist, use_container_width=True)

    # -----------------------------------------------------------------------------
    # 4. SIMULADOR IA v3 (CON DATOS ESCRITOS)
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.header("🔮 Simulador Predictivo (v3)")
    
    model_path = os.path.join(os.path.dirname(__file__), '../modelos/modelo_energia_v3.pkl')
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Configuración")
                sector_p = st.selectbox("🏢 Sector", ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"])
                occ_f = st.slider("Ocupación (%)", 0, 100, 60)
                temp_f = st.slider("Clima Previsto (°C)", 5, 35, 17)
            
            with c2:
                sede_idx = list(df['sede'].unique()).index(sede_selec)
                sector_idx = ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"].index(sector_p)
                
                horas = list(range(24))
                preds = []
                for h in horas:
                    input_row = pd.DataFrame([[h, 1, 10, sede_idx, sector_idx, occ_f, temp_f]], 
                                            columns=['hora', 'dia_semana', 'mes', 'sede_n', 'sector_n', 'ocupacion_pct', 'temperatura_exterior_c'])
                    preds.append(model.predict(input_row)[0])
                
                # --- NUEVA SECCIÓN: VALORES ESCRITOS ---
                st.subheader("📊 Métricas de la Proyección")
                m1, m2, m3 = st.columns(3)
                
                valor_max = max(preds)
                valor_min = min(preds)
                total_dia = sum(preds)
                
                m1.metric("Consumo Pico", f"{valor_max:.2f} kWh", help="El valor más alto predicho en el día")
                m2.metric("Consumo Base", f"{valor_min:.2f} kWh", help="El valor más bajo predicho (consumo constante)")
                m3.metric("Total 24h Est.", f"{total_dia:.2f} kWh", delta=f"{(total_dia/24):.2f} avg/h")

                # Gráfica
                fig_pred = px.area(x=horas, y=preds, title=f"Curva de Carga Predicha: {sector_p}",
                                   labels={'x': 'Hora del día', 'y': 'Energía (kWh)'}, 
                                   color_discrete_sequence=['#F1C40F'])
                st.plotly_chart(fig_pred, use_container_width=True)

                # Tabla detallada (opcional, por si quieren ver la lista)
                with st.expander("🔎 Ver desglose hora por hora"):
                    df_detalles = pd.DataFrame({"Hora": horas, "Predicción (kWh)": preds})
                    st.dataframe(df_detalles.set_index("Hora"), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error en el Simulador: {e}")
    else:
        st.warning("No se encontró el archivo 'modelo_energia_v3.pkl' en la carpeta modelos.")
