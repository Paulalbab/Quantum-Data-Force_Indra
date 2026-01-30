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
    # 4. SIMULADOR IA v3 (CON SELECCIÓN DE HORA Y DÍA)
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.header("🔮 Simulador Predictivo Personalizado (v3)")
    
    model_path = os.path.join(os.path.dirname(__file__), '../modelos/modelo_energia_v3.pkl')
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Configuración del Escenario")
                sector_p = st.selectbox("🏢 Sector", ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"])
                
                # NUEVO: Selección de Día y Hora específica
                dia_p = st.select_slider("📅 Día de la semana", 
                                        options=[0, 1, 2, 3, 4, 5, 6],
                                        value=1,
                                        format_func=lambda x: ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"][x])
                
                hora_p = st.slider("⏰ Hora específica para consulta", 0, 23, 10)
                
                st.markdown("---")
                occ_f = st.slider("Ocupación Esperada (%)", 0, 100, 60)
                temp_f = st.slider("Clima Previsto (°C)", 5, 35, 17)
            
            with c2:
                sede_idx = list(df['sede'].unique()).index(sede_selec)
                sector_idx = ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"].index(sector_p)
                
                # 1. Predicción puntual para la hora y día seleccionado
                input_puntual = pd.DataFrame([[hora_p, dia_p, 10, sede_idx, sector_idx, occ_f, temp_f]], 
                                            columns=['hora', 'dia_semana', 'mes', 'sede_n', 'sector_n', 'ocupacion_pct', 'temperatura_exterior_c'])
                pred_puntual = model.predict(input_puntual)[0]

                # 2. Generar datos para la curva de 24 horas del día elegido
                horas = list(range(24))
                preds_24h = []
                for h in horas:
                    input_row = pd.DataFrame([[h, dia_p, 10, sede_idx, sector_idx, occ_f, temp_f]], 
                                            columns=['hora', 'dia_semana', 'mes', 'sede_n', 'sector_n', 'ocupacion_pct', 'temperatura_exterior_c'])
                    preds_24h.append(model.predict(input_row)[0])
                
                # --- MÉTRICAS EN TEXTO ---
                st.subheader(f"📍 Predicción Puntual ({hora_p}:00)")
                
                # Mostramos el dato grande y llamativo
                st.metric(label=f"Consumo estimado en {sector_p}", 
                          value=f"{pred_puntual:.2f} kWh",
                          delta=f"{(pred_puntual - (sum(preds_24h)/24)):+.2f} vs promedio día")

                # Tarjetas de apoyo
                m1, m2 = st.columns(2)
                m1.metric("Pico del día", f"{max(preds_24h):.2f} kWh")
                m2.metric("Total día estimado", f"{sum(preds_24h):.2f} kWh")

                # Gráfica interactiva
                fig_pred = px.area(x=horas, y=preds_24h, 
                                   title=f"Curva de Carga para el día seleccionado",
                                   labels={'x': 'Hora del día', 'y': 'Energía (kWh)'}, 
                                   color_discrete_sequence=['#F1C40F'])
                
                # Añadir una línea vertical roja en la hora seleccionada para que se vea claro
                fig_pred.add_vline(x=hora_p, line_dash="dash", line_color="red", 
                                  annotation_text=f"Consulta: {hora_p}:00")
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error en el Simulador: {e}")
    else:
        st.warning("No se encontró el archivo 'modelo_energia_v3.pkl'.")
