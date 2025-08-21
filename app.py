import streamlit as st
import requests
import pandas as pd
import joblib
import altair as alt
import folium
from streamlit_folium import st_folium
from datetime import date, timedelta

# === Configuración de página ===
st.set_page_config(
    page_title="Predicción Eólica Chile",
    page_icon="⚡",
    layout="wide"
)

# === CSS estético y dashboard ===
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
    padding: 20px;
}
h1, h2, h3 {
    font-family: 'Helvetica Neue', sans-serif;
    font-weight: 600;
    color: #0d3b66;
}
.stDataFrame {
    background: white;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
.dashboard-card {
    border-radius: 12px;
    padding: 10px;
    text-align:center;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    margin-bottom:10px;
}
.dashboard-card h4 {
    margin:0;
    font-size:16px;
    color:#0d3b66;
}
.dashboard-card p {
    font-size:22px;
    font-weight:bold;
    margin:5px 0 0 0;
}
</style>
""", unsafe_allow_html=True)

# === Cargar modelo ===

@st.cache_resource 
def load_model():
    model = joblib.load('random_forest_eolicos_todos.pkl.gz')
    return model

model = load_model()


# === Título ===
st.title("⚡ Predicción de Generación Eólica en Chile")
st.markdown(
    "Selecciona una ubicación en el mapa y un rango de fechas para obtener el pronóstico horario de "
    "generación eólica basado en la API de **Open-Meteo** y tu modelo entrenado."
)

# === Estado inicial del marker ===
if "marker_location" not in st.session_state:
    st.session_state.marker_location = [-33.45, -70.66]  # default
if "zoom" not in st.session_state:
    st.session_state.zoom = 5
if "loading_map" not in st.session_state:
    st.session_state.loading_map = False

# === Selector de fechas ===
st.sidebar.header("📅 Parámetros de Pronóstico")
today = date.today()
start_date = st.sidebar.date_input("Fecha de inicio", today)
end_date = st.sidebar.date_input("Fecha de término", today + timedelta(days=2))
if (end_date - start_date).days > 16:
    st.sidebar.error("⚠️ El rango máximo permitido es de 16 días.")
    end_date = start_date + timedelta(days=16)

# === Layout principal ===
col1, col2 = st.columns([1.2, 2])

# --- Columna mapa ---
with col1:
    st.subheader("🌍 Selección en el mapa")
    loading_container = st.empty()

    # Crear mapa base con marker
    m = folium.Map(location=st.session_state.marker_location, zoom_start=st.session_state.zoom)
    folium.Marker(
        location=st.session_state.marker_location,
        draggable=False
    ).add_to(m)

    # Mostrar mapa y capturar clic
    map_data = st_folium(m, width=480, height=500, key="folium_map")

    # Actualizar marker inmediatamente al hacer click
    if map_data and map_data.get("last_clicked"):
        lat, lng = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        if [lat, lng] != st.session_state.marker_location:
            st.session_state.marker_location = [lat, lng]
            st.session_state.zoom = map_data["zoom"]
            st.session_state.loading_map = True
            # Redibujar mapa con nueva ubicación
            m = folium.Map(location=st.session_state.marker_location, zoom_start=st.session_state.zoom)
            folium.Marker(
                location=st.session_state.marker_location,
                draggable=False
            ).add_to(m)
            st_folium(m, width=480, height=500, key="folium_map")

# --- Columna resultados ---
with col2:
    st.subheader("📊 Resultados")
    if st.session_state.loading_map:
        loading_container.info("⏳ Cargando datos para la nueva ubicación...")
        st.session_state.loading_map = False

    lat, lon = st.session_state.marker_location

    with st.spinner("⏳ Obteniendo pronóstico y generando predicciones..."):
        # === API Open-Meteo ===
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,relative_humidity_2m,rain,cloud_cover,"
            "wind_speed_10m,wind_speed_100m,direct_radiation"
            f"&start_date={start_date}&end_date={end_date}&timezone=auto"
        )
        response = requests.get(url)
        data = response.json()

        hourly = data["hourly"]
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        df["hora"] = df["time"].dt.hour
        df["lat"] = lat
        df["lon"] = lon

        df = df.rename(columns={
            "temperature_2m": "temperature_2m (°C)",
            "relative_humidity_2m": "relative_humidity_2m (%)",
            "rain": "rain (mm)",
            "cloud_cover": "cloud_cover (%)",
            "wind_speed_10m": "wind_speed_10m (m/s)",
            "wind_speed_100m": "wind_speed_100m (m/s)",
            "direct_radiation": "direct_radiation (W/m²)"
        })

        features = [
            "temperature_2m (°C)", "relative_humidity_2m (%)", "rain (mm)",
            "cloud_cover (%)", "wind_speed_10m (m/s)", "wind_speed_100m (m/s)",
            "direct_radiation (W/m²)", "hora", "lat", "lon"
        ]
        df["prediction (mW/h)"] = model.predict(df[features])

        # --- Tarjetas dashboard ---
        total_gen = df["prediction (mW/h)"].sum()
        max_gen = df["prediction (mW/h)"].max()
        avg_wind = df["wind_speed_10m (m/s)"].mean()

        st.markdown(f"""
        <div style="display:flex; gap:15px; margin-bottom:15px;">
            <div class="dashboard-card" style="background-color:#e0f3f8; flex:1;">
                <h4>Generación Total</h4>
                <p>{total_gen:,.0f} mW/h</p>
            </div>
            <div class="dashboard-card" style="background-color:#fce8e6; flex:1;">
                <h4>Máx. Generación Horaria</h4>
                <p>{max_gen:.0f} mW/h</p>
            </div>
            <div class="dashboard-card" style="background-color:#e8f7e4; flex:1;">
                <h4>Vel. Viento Promedio</h4>
                <p>{avg_wind:.1f} m/s</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Dataframe ---
        st.dataframe(df[["time", "prediction (mW/h)"]])

        # --- Gráfico con gradiente ---
        chart = alt.Chart(df).mark_area(
            line={"color": "#0d3b66"},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color="#0d3b66", offset=0),
                       alt.GradientStop(color="#a9cfe7", offset=1)],
                x1=1, y1=1, x2=1, y2=0
            ),
            opacity=0.5
        ).encode(
            x=alt.X("time:T", title="Tiempo"),
            y=alt.Y("prediction (mW/h):Q", title="Generación Eólica (mW/h)")
        ).properties(
            title=f"Predicción de Generación Eólica ({start_date} → {end_date})",
            width=700,
            height=350
        )

        st.altair_chart(chart, use_container_width=True)

    loading_container.empty()
