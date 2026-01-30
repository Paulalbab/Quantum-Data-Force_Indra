import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN
st.set_page_config(page_title="Quantum Data Force | UPTC", page_icon="⚡", layout="wide")

st.title(" Monitor Energético Histórico y Predictivo - UPTC")
st.markdown("**Equipo:** Quantum Data Force | Análisis de Tendencias 2018-2025")

# -----------------------------------------------------------------------------
# 2. CARGA DE DATOS
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, '../datos/consumos_uptc.zip')
    try:
        df = pd.read_csv(file_path, compression='zip')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

df = load_data()

if df is not None:
    st.sidebar.header("🔍 Control de Visualización")
    sede_selec = st.sidebar.selectbox("Selecciona Sede:", df['sede'].unique())
    
    # Agrupación para la gráfica histórica (Día o Mes)
    agrupar = st.sidebar.radio("Ver tendencia por:", ["Día", "Mes"])
    
    df_sede = df[df['sede'] == sede_selec].copy()
    
    # -----------------------------------------------------------------------------
    # 3. TENDENCIA HISTÓRICA TOTAL
    st.subheader(f" Línea de Tiempo Completa: Sede {sede_selec}")
    
    if agrupar == "Día":
        df_plot = df_sede.resample('D', on='timestamp').mean().reset_index()
    else:
        df_plot = df_sede.resample('M', on='timestamp').mean().reset_index()

    fig_total = px.line(df_plot, x='timestamp', y='energia_total_kwh', 
                        title=f"Evolución del Consumo (Agrupado por {agrupar})",
                        color_discrete_sequence=['#2E86C1'],
                        labels={'energia_total_kwh': 'Consumo Promedio (kWh)', 'timestamp': 'Fecha'})
    
    fig_total.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_total, use_container_width=True)

    # -----------------------------------------------------------------------------
    # 4. SIMULADOR DE PREDICCIÓN IA
    st.markdown("---")
    st.header(" Simulador de Proyección IA")
    
    model_path = os.path.join(os.path.dirname(__file__), '../modelos/modelo_energia_v3.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            sector_p = st.selectbox(" Sector", ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"])
            occ_f = st.slider("Ocupación (%)", 0, 100, 50)
            temp_f = st.slider("Temperatura (°C)", 5, 35, 18)
            
        with col2:
            # Lógica de predicción 24h
            sede_idx = list(df['sede'].unique()).index(sede_selec)
            sector_idx = ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"].index(sector_p)
            
            horas = list(range(24))
            preds = [model.predict(pd.DataFrame([[h, 1, 10, sede_idx, sector_idx, occ_f, temp_f]], 
                     columns=['hora', 'dia_semana', 'mes', 'sede_n', 'sector_n', 'ocupacion_pct', 'temperatura_exterior_c']))[0] for h in horas]
            
            fig_p = px.area(x=horas, y=preds, title=f"Predicción para un día típico en {sector_p}",
                           labels={'x': 'Hora del día', 'y': 'kWh'}, color_discrete_sequence=['#F39C12'])
            st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.warning("Sube el modelo V3 para ver las predicciones.")
