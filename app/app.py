import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PAG
st.set_page_config(
    page_title="Quantum Data Force | UPTC",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #F1C40F; }
    [data-testid="stMetricDelta"] { font-size: 16px; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. FUNCIONES DE CARGA  
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, '../datos/consumos_uptc.zip')
    try:
        df = pd.read_csv(file_path, compression='zip')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['sede_n'] = df['sede'].astype('category').cat.codes
        return df
    except Exception as e:
        st.error(f"Error al cargar base de datos: {e}")
        return None

@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, '../modelos/modelo_energia_v4.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

df = load_data()
model = load_model()

if df is not None:
    st.title("Sistema de Inteligencia Energética - UPTC")
    st.markdown(f"**Análisis Histórico 2018-2025** | Equipo: **Quantum Data Force**")

    # --- BARRA LATERAL (FILTROS) ---
    st.sidebar.header(" Panel de Control")
    sede_selec = st.sidebar.selectbox(" Selecciona la Sede:", df['sede'].unique())
    vista_h = st.sidebar.radio("Resolución de tendencia histórica:", ["Mensual", "Diaria"])
    
    df_sede = df[df['sede'] == sede_selec].copy()

    # -----------------------------------------------------------------------------
    # 3. VISUALIZACIÓN HISTÓRICA (2018 - 2025)
    st.subheader(f" Análisis de Tendencia Total: {sede_selec}")
    
    rule = 'M' if vista_h == "Mensual" else 'D'
    df_hist = df_sede.set_index('timestamp').select_dtypes(include=['number']).resample(rule).mean().reset_index()

    fig_h = px.line(df_hist, x='timestamp', y='energia_total_kwh', 
                    title=f"Consumo Promedio ({vista_h})",
                    color_discrete_sequence=['#1ABC9C'],
                    labels={'energia_total_kwh': 'Consumo (kWh)', 'timestamp': 'Año'})
    fig_h.update_xaxes(rangeslider_visible=True) # Selector de rango de años
    st.plotly_chart(fig_h, use_container_width=True)

    # -----------------------------------------------------------------------------
    # 4. PREDICCIÓN
    st.markdown("---")
    st.header(" Simulador de Predicción IA")
    st.markdown("Proyección detallada por sector, lugar y tiempo utilizando el Modelo v4.")

    if model is not None:
        try:
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.subheader("Configuración del Escenario")
                sectores = ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"]
                sector_p = st.selectbox("Sector Específico", sectores)
                
                # Selector de Fecha
                fecha_p = st.date_input(" Selecciona el día a simular", value=pd.to_datetime("2025-06-10"))
                
                # Selector de Hora 
                hora_p = st.slider("Consultar hora específica", 0, 23, 10)
                
                # Variables ambientales
                occ_f = st.slider("Nivel de Ocupación (%)", 0, 100, 75)
                temp_f = st.slider("Clima Estimado (°C)", 5, 35, 16)

            with c2:
                # Preparación de variables para la IA
                sede_idx = list(df['sede'].unique()).index(sede_selec)
                sector_idx = sectores.index(sector_p)
                dia_w = fecha_p.weekday()
                mes_p = fecha_p.month
                
                # A. Predicción para la curva de 24 horas
                horas = list(range(24))
                preds_24h = []
                for h in horas:
                    
                    input_row = pd.DataFrame([[h, dia_w, mes_p, sede_idx, sector_idx, occ_f, temp_f]], 
                                            columns=['hora', 'dia_semana', 'mes', 'sede_n', 'sector_n', 'ocupacion_pct', 'temperatura_exterior_c'])
                    preds_24h.append(model.predict(input_row)[0])
                
                # B. Métricas Escritas (Datos exactos)
                st.subheader(f" Resultados: {sector_p} ({sede_selec})")
                m1, m2, m3 = st.columns(3)
                
                dato_puntual = preds_24h[hora_p]
                consumo_pico = max(preds_24h)
                total_dia = sum(preds_24h)
                
                m1.metric(f"Consumo a las {hora_p}:00", f"{dato_puntual:.2f} kWh")
                m2.metric("Pico Proyectado", f"{consumo_pico:.2f} kWh")
                m3.metric("Total día estimado", f"{total_dia:.2f} kWh")

                # C. Gráfica de la simulación
                fig_p = px.area(x=horas, y=preds_24h, 
                                title=f"Curva de Carga Proyectada para el {fecha_p}",
                                labels={'x': 'Hora del día', 'y': 'Energía (kWh)'},
                                color_discrete_sequence=['#F1C40F'])
                
                # Línea indicadora de la hora consultada
                fig_p.add_vline(x=hora_p, line_dash="dash", line_color="#E74C3C", 
                               annotation_text=f"Consulta: {hora_p}:00")
                
                st.plotly_chart(fig_p, use_container_width=True)

        except Exception as e:
            st.error(f"Error en la predicción: {e}")
            st.info("Asegúrate de que el modelo v4 esté correctamente cargado.")
    else:
        st.warning(" No se detectó el archivo 'modelo_energia_v4.pkl' en la carpeta /modelos.")

else:
    st.error(" No se pudo conectar con la base de datos en /datos/consumos_uptc.zip")
