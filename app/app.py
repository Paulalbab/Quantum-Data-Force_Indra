import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- 1. CONFIGURACIÓN DE LA PÁGINA 
st.set_page_config(page_title="Quantum Data Force", layout="wide", page_icon="⚡")

st.title("⚡ Monitor Energético UPTC - IA Minds 2026")
st.markdown("**Estado del Sistema:** 🟢 En línea | **Fuente de Datos:** Repositorio Seguro")

# cargar zip a csv
@st.cache_data
def cargar_datos():
    ruta_zip = os.path.join(os.path.dirname(__file__), '../datos/consumos_uptc.zip')
    
    try:
        df = pd.read_csv(ruta_zip, compression='zip')
        
        df['timestamp'] = pd.to_datetime(df['timestamp']) # Texto a Fecha
        df = df[df['energia_total_kwh'] >= 0]             # Borrar negativos
        df['co2_kg'] = df['co2_kg'].fillna(0)             # Rellenar vacíos
        
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

# Cargar los datos
df = cargar_datos()

if not df.empty:
    
    # Filtros
    st.sidebar.header("Filtros")
    sede = st.sidebar.selectbox("Selecciona Sede:", df['sede'].unique())
    
    # Filtrar datos
    df_sede = df[df['sede'] == sede]
    
    # KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Energía (Histórico)", f"{df_sede['energia_total_kwh'].sum():,.0f} kWh")
    kpi2.metric("Pico Máximo Detectado", f"{df_sede['energia_total_kwh'].max():.2f} kWh")
    kpi3.metric("Registros Analizados", f"{len(df_sede)}")
    
    # Gráfica Principal
    st.subheader(f"Comportamiento Energético: {sede}")
    # Tse toma el ultimo mes
    ultimo_mes = df_sede[df_sede['timestamp'] > df_sede['timestamp'].max() - pd.Timedelta(days=30)]
    
    fig = px.line(ultimo_mes, x='timestamp', y='energia_total_kwh', 
                  title="Últimos 30 días de consumo", color_discrete_sequence=['#00CC96'])
    st.plotly_chart(fig, use_container_width=True)

    st.success("¡Conexión exitosa! El archivo ZIP se leyó correctamente.")

else:
    st.warning("Esperando conexión con los datos...")
