import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib

st.set_page_config(page_title="Quantum Data Force | UPTC", page_icon="⚡", layout="wide")

# CARGA DE DATOS
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, '../datos/consumos_uptc.zip')
    try:
        df = pd.read_csv(file_path, compression='zip')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Generar códigos de sede consistentes con el entrenamiento
        df['sede_n'] = df['sede'].astype('category').cat.codes
        return df
    except: return None

df = load_data()

if df is not None:
    st.title("⚡ Dashboard Energético UPTC (2018-2025)")
    
    # SIDEBAR
    sede_selec = st.sidebar.selectbox("Selecciona Sede:", df['sede'].unique())
    df_sede = df[df['sede'] == sede_selec]

    # GRÁFICA HISTÓRICA TOTAL
    st.subheader(f"📈 Tendencia Histórica: {sede_selec}")
    df_hist = df_sede.set_index('timestamp').resample('M').mean().reset_index()
    fig_h = px.line(df_hist, x='timestamp', y='energia_total_kwh', color_discrete_sequence=['#1ABC9C'])
    fig_h.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_h, use_container_width=True)

    # SIMULADOR IA v3
    st.markdown("---")
    st.header(" Simulador de Predicción Detallado")
    
    model_path = os.path.join(os.path.dirname(__file__), '../modelos/modelo_energia_v3(2).pkl')
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Configuración")
                sectores = ["Comedores", "Salones", "Laboratorios", "Auditorios", "Oficinas"]
                sector_p = st.selectbox("Sector", sectores)
                
                # SELECCIÓN DE DÍA Y HORA
                fecha_p = st.date_input(" Día a predecir", value=pd.to_datetime("2025-05-15"))
                hora_p = st.slider("Hora a consultar", 0, 23, 10)
                
                occ_f = st.slider("Ocupación (%)", 0, 100, 70)
                temp_f = st.slider("Clima (°C)", 5, 35, 16)

            with c2:
                # Mapeo de índices
                sede_idx = list(df['sede'].unique()).index(sede_selec)
                sector_idx = sectores.index(sector_p)
                dia_semana = fecha_p.weekday()
                mes_p = fecha_p.month
                
                # Predicción puntual
                input_pt = pd.DataFrame([[hora_p, dia_semana, mes_p, sede_idx, sector_idx, occ_f, temp_f]], 
                                       columns=['hora', 'dia_semana', 'mes', 'sede_n', 'sector_n', 'ocupacion_pct', 'temperatura_exterior_c'])
                res_pt = model.predict(input_pt)[0]

                # Curva 24h
                horas = list(range(24))
                preds_24h = []
                for h in horas:
                    row = pd.DataFrame([[h, dia_semana, mes_p, sede_idx, sector_idx, occ_f, temp_f]], 
                                      columns=['hora', 'dia_semana', 'mes', 'sede_n', 'sector_n', 'ocupacion_pct', 'temperatura_exterior_c'])
                    preds_24h.append(model.predict(row)[0])

                # MOSTRAR DATOS ESCRITOS
                st.subheader(f"Resultado para {fecha_p} a las {hora_p}:00")
                st.metric("Consumo Predicho", f"{res_pt:.2f} kWh")
                
                # Gráfica
                fig_p = px.area(x=horas, y=preds_24h, title="Proyección 24 Horas",
                               labels={'x': 'Hora', 'y': 'kWh'}, color_discrete_sequence=['#F1C40F'])
                fig_p.add_vline(x=hora_p, line_dash="dash", line_color="red")
                st.plotly_chart(fig_p, use_container_width=True)

        except Exception as e: st.error(f"Error de lógica: {e}")
    else: st.warning("Sube el modelo_energia_v3(2).pkl")
