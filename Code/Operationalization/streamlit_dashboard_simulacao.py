"""
Dashboard Streamlit para:
1. Simulação de arremessos do Kobe Bryant com entrada manual de dados;
2. Visualização espacial dos arremessos históricos com heatmap e pontos coloridos.

Funcionalidades:
- Entrada de dados interativa para prever acerto ou erro de arremesso;
- Visualização de mapa com base em latitude/longitude;
- Histórico de simulações salvas localmente;
- Registro de simulações e gráficos no MLflow.
"""

import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from datetime import datetime
import os
import mlflow
import tempfile
from pycaret.classification import load_model
from sklearn.metrics import log_loss, f1_score

# Configuração da página
st.set_page_config(page_title="📊 Dashboard - Simulação de Arremessos - Modelo Kobe Bryant", layout="wide")

# Navegação por abas
aba = st.sidebar.selectbox(
    "Selecione a visualização:",
    ["Simulação", "Mapa de Arremessos"]
)

# -----------------------------
# ABA: SIMULAÇÃO INTERATIVA
# -----------------------------
if aba == "Simulação":
    st.title("🏀 Simulador de Arremessos - Kobe Bryant")

    # Carrega modelo treinado
    model = load_model("../../Data/Modeling/modelo_final")

    st.sidebar.header("🎛️ Simule uma Jogada")

    # Inputs do usuário
    lat = st.sidebar.slider("Latitude (lat)", min_value=33.5, max_value=34.0, step=0.01, value=33.93)
    lon = st.sidebar.slider("Longitude (lon)", min_value=-118.5, max_value=-118.0, step=0.01, value=-118.05)
    minutes = st.sidebar.slider("Minutos Restantes", min_value=0, max_value=11, step=1, value=5)
    period = st.sidebar.slider("Período", min_value=1, max_value=7, step=1, value=2)
    playoffs = st.sidebar.selectbox("É Playoffs?", options=[0, 1])
    distance = st.sidebar.slider("Distância do Arremesso", min_value=0, max_value=50, step=1, value=18)

    input_data = pd.DataFrame({
        "lat": [lat],
        "lon": [lon],
        "minutes_remaining": [minutes],
        "period": [period],
        "playoffs": [playoffs],
        "shot_distance": [distance],
    })

    if st.sidebar.button("🏹 Avaliar Arremesso"):
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        st.subheader("🎯 Resultado da Jogada")
        resultado = "✅ Acerto" if pred == 1 else "❌ Erro"
        cor = "green" if pred == 1 else "red"
        st.markdown(f"**Classificação:** <span style='color:{cor}'>{resultado}</span>", unsafe_allow_html=True)
        st.metric("Probabilidade de Acerto", f"{proba*100:.2f}%")

        st.markdown("---")
        st.markdown("**Variáveis usadas na simulação:**")
        st.dataframe(input_data)

        # Salva localmente simulação
        log_path = "../../Data/Logs"
        os.makedirs(log_path, exist_ok=True)
        log_file = os.path.join(log_path, "simulacoes.csv")

        log_data = input_data.copy()
        log_data["prediction"] = pred
        log_data["proba"] = proba
        log_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if os.path.exists(log_file):
            log_data.to_csv(log_file, mode="a", header=False, index=False)
        else:
            log_data.to_csv(log_file, index=False)

        st.markdown("---")
        st.subheader("📜 Histórico de Simulações")
        historico = pd.read_csv(log_file)
        st.dataframe(historico.tail(10))

        # Registro no MLflow
        mlflow.set_experiment("PipelineAplicacao")
        with mlflow.start_run(run_name="StreamlitSimulacao"):
            mlflow.log_param("lat", lat)
            mlflow.log_param("lon", lon)
            mlflow.log_param("minutes_remaining", minutes)
            mlflow.log_param("period", period)
            mlflow.log_param("playoffs", playoffs)
            mlflow.log_param("shot_distance", distance)

            mlflow.log_metric("proba", proba)
            mlflow.log_metric("prediction", int(pred))

            # Salva simulação como artefato temporário
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
                temp_path = tmp.name
                log_data.to_csv(temp_path, index=False)

            mlflow.log_artifact(temp_path, artifact_path="simulacoes")
            os.remove(temp_path)

# -----------------------------
# ABA: MAPA DE ARREMESSOS
# -----------------------------
elif aba == "Mapa de Arremessos":
    st.title("📍 Localização dos Arremessos - Kobe Bryant")

    # Caminhos dos arquivos
    quadra_path = '../../Docs/Imagens/charlotte_key_zone.jpeg'
    dados_path = '../../Data/Raw/dataset_kobe_dev.parquet'

    try:
        img = Image.open(quadra_path)
        df = pd.read_parquet(dados_path)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # Mapeamento para visualização
    mapa_valores = {1: 'Cesta', 0: 'Erro', None: 'Desconhecido'}
    df['resultado'] = df['shot_made_flag'].map(mapa_valores)
    df['cor'] = df['shot_made_flag'].map({1: 'green', 0: 'red'}).fillna('black')

    # Filtros laterais
    st.sidebar.header("Filtros de Visualização")

    resultado = st.sidebar.multiselect(
        'Resultado do arremesso:',
        options=['Cesta', 'Erro', 'Desconhecido'],
        default=['Cesta', 'Erro', 'Desconhecido']
    )

    distancia_max = st.sidebar.slider(
        "Distância máxima do arremesso (ft):",
        min_value=0, max_value=50, value=50, step=1
    )

    tipo_visu = st.sidebar.radio(
        "Tipo de visualização:",
        ["Pontos coloridos", "Heatmap"]
    )

    # Aplica filtros
    df_filtrado = df[
        (df['resultado'].isin(resultado)) &
        (df['shot_distance'] <= distancia_max)
    ]

    if df_filtrado.empty:
        st.warning("Nenhum dado disponível com os filtros selecionados.")
        st.stop()

    # Criação do gráfico
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-118.5, -118.05)
    ax.set_ylim(33.5, 34.1)
    ax.imshow(img, extent=[-118.5, -118.05, 33.5, 34.1], aspect='auto', zorder=0)

    if tipo_visu == "Pontos coloridos":
        ax.scatter(df_filtrado['lon'], df_filtrado['lat'], c=df_filtrado['cor'],
                   s=10, alpha=0.7, edgecolors='k', linewidths=0.1, zorder=1)
        ax.set_title("Localização dos Arremessos - Kobe Bryant")
        legenda = [
            Patch(color='green', label='Cesta'),
            Patch(color='red', label='Erro'),
            Patch(color='black', label='Desconhecido')
        ]
        ax.legend(handles=legenda, loc='lower left', title='Resultado do Arremesso')
    else:
        import seaborn as sns
        ax.set_title("Zonas Quentes de Arremesso")
        sns.kdeplot(
            x=df_filtrado['lon'],
            y=df_filtrado['lat'],
            fill=True,
            cmap="hot",
            bw_adjust=0.5,
            thresh=0.05,
            levels=100,
            alpha=0.8,
            ax=ax
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig)
