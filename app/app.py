import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y ESTILO
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quantum Data Force | UPTC", page_icon="⚡", layout="wide")

st.title("⚡ Inteligencia Energética UPTC (2018 - 2025)")
st.markdown("**Análisis Avanzado:** Tendencias Históricas y Proyecciones con Modelo IA v3")

# -----------------------------------------------------------------------------
# 2. CARGA DE DATOS (Optimizada para grandes volúmenes)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, '../datos/consumos_uptc.zip')
    try:
        df = pd.read_csv(file_path, compression='zip')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error al cargar la base de datos: {e}")
        return None

df = load_data()

if df is not None:
    # --- BARRA LATERAL ---
    st.sidebar.header("🕹️ Panel de Control")
    sede_selec = st.sidebar.selectbox("Selecciona Sede:", df['sede'].unique())
    
    # Opción para suavizar la curva de 8 años
    vista = st.sidebar.radio("Resolución de la tendencia:", ["Diaria", "Mensual (Recomendado)"])
    
    df_sede = df[df['sede'] == sede_selec].copy()
    
    # -----------------------------------------------------------------------------
    # 3. TENDENCIA HISTÓRICA TOTAL (2018-2025)
    # -----------------------------------------------------------------------------
    st.subheader(f"📈 Análisis de Ciclos Energéticos: {sede_selec}")
    
    # Agrupamos para que la gráfica sea legible y no pese demasiado
    resample_rule = 'D' if vista == "Diaria" else 'M'
    df_hist = df_sede.set_index('timestamp').resample(resample_rule).mean().reset_index()

    fig_hist = px.line(df_hist, x='timestamp', y='energia_total_kwh', 
                        title=f"Evolución Histórica Agrupada por {vista}",
                        color_discrete_sequence=['#1ABC9C'],
                        labels={'energia_total_kwh': 'Consumo (kWh)', 'timestamp': 'Año'})
    
    # Añadimos el Range Slider para que el jurado navegue entre 2018 y 2025
    fig_hist.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_hist, use_container_width=True)

    # -----------------------------------------------------------------------------
    # 4. SIMULADOR CON MODELO V3
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.header("🔮 Simulador de Proyección IA (v3)")
    
    # Cambiamos la ruta para que busque el modelo v3
    model_path = os.path.join(os.path.dirname(__file__), '../modelos/modelo_energia_v3.pkl')
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.info("Parámetros de Simulación")
            sector_p = st.selectbox("🏢 Sector", ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"])
            occ_f = st.slider("Ocupación (%)", 0, 100, 60)
            temp_f = st.slider("Temperatura Prevista (°C)", 5, 35, 17)
            
        with c2:
            # Procesamiento para el modelo v3
            sede_idx = list(df['sede'].unique()).index(sede_selec)
            sector_idx = ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"].index(sector_p)
            
            # Generar curva de 24 horas predicha
            horas = list(range(24))
            # Ajusta las columnas según el orden que usaste en tu Colab para el v3
            preds = []
            for h in horas:
                input_row = pd.DataFrame([[h, 1, 10, sede_idx, sector_idx, occ_f, temp_f]], 
                                        columns=['hora', 'dia_semana', 'mes', 'sede_n', 'sector_n', 'ocupacion_pct', 'temperatura_exterior_c'])
                preds.append(model.predict(input_row)[0])
            
            fig_pred = px.area(x=horas, y=preds, title=f"Comportamiento Predicho: {sector_p}",
                               labels={'x': 'Hora', 'y': 'Consumo (kWh)'},
                               color_discrete_sequence=['#F1C40F'])
            st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.warning("⚠️ El archivo 'modelo_energia_v3.pkl' no se encuentra en la carpeta modelos.")
