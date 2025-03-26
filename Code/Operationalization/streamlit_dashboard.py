import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

sns.set_style("whitegrid")

st.set_page_config(page_title="Dashboard - Modelo Kobe Bryant", layout="wide")
st.title("📊 Dashboard Analítico - Modelo Kobe Bryant")

# Caminho para modelo e predições
caminho_predicoes = "../../Data/Processed/predictions_prod.parquet"
caminho_modelo = "../../Data/Modeling/modelo_final"

# Carrega o modelo
model = load_model(caminho_modelo)

# Layout principal com colunas
col1, col2 = st.columns([1, 2])

with col1:
    st.sidebar.title("📘 Sobre este Dashboard")
    st.sidebar.markdown("Visualização e avaliação do desempenho do modelo de classificação de arremessos do Kobe Bryant.")

    st.sidebar.markdown("""
    **Como usar:**
    - ✅ Execute o pipeline de aplicação
    - 📊 Este painel analisará a produção atual
    - 🏀 Para simulação de novas jogadas, use o arquivo:

    ```bash
    streamlit run Code/Operationalization/streamlit_dashboard_simulacao.py
    ```
    """)

with col2:
    st.markdown("""
    ### Avaliação Analítica do Modelo
    O modelo foi treinado com 16.228 registros e avaliado com 4.057 dados de produção. Esta interface permite:

    - Ver amostras reais com predições
    - Avaliar métricas como Accuracy, F1, Recall, LogLoss
    - Visualizar a matriz de confusão
    - Explorar a distribuição de probabilidades previstas
    """)

    # Verifica se o arquivo de predições existe
    if os.path.exists(caminho_predicoes):
        df = pd.read_parquet(caminho_predicoes)

        st.subheader("▶️ Amostra das predições da produção")
        st.dataframe(df.head(10))

        # Se existir a coluna de flag (real)
        if "shot_made_flag" in df.columns:
            st.subheader("📊 Avaliação do Modelo")
            y_true = df["shot_made_flag"].dropna()
            y_pred = df.loc[y_true.index, "prediction"]

            report = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))

            # Matriz de confusão
            st.subheader("🔢 Matriz de Confusão")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Erro", "Acerto"],
                        yticklabels=["Erro", "Acerto"], ax=ax)
            plt.xlabel("Predito")
            plt.ylabel("Real")
            st.pyplot(fig)

            # ------------------------------------------------------------------
            # Seção comparativa estilo "faça algo assim"
            # ------------------------------------------------------------------
            st.subheader("📈 Comparativo de Arremessos do Kobe")

            # Calcula métricas simples
            total_arremessos = df.shape[0]
            acertos_previstos = (df["prediction"] == 1).sum()
            taxa_acerto = (acertos_previstos / total_arremessos) * 100

            # Organiza em duas colunas: métricas (esquerda) e gráfico (direita)
            met_col, chart_col = st.columns([1, 2])

            # 1) Coluna de métricas
            with met_col:
                st.metric("Total de Arremessos", total_arremessos)
                st.metric("Acertos Previstos", acertos_previstos)
                st.metric("Taxa de Acerto (%)", f"{taxa_acerto:.2f}%")

            # 2) Coluna do gráfico de barras
            with chart_col:
                # Monta um DataFrame para o gráfico de barras
                data_bar = {
                    "Tipo": ["Acertos Previstos", "Erros Previstos"],
                    "Quantidade": [acertos_previstos, total_arremessos - acertos_previstos]
                }
                df_bar = pd.DataFrame(data_bar)

                fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                sns.barplot(
                    data=df_bar,
                    x="Tipo",
                    y="Quantidade",
                    palette="Blues_r"
                )
                ax_bar.set_title("Acertos vs. Erros (Preditos)")
                ax_bar.set_xlabel("")
                ax_bar.set_ylabel("Quantidade")
                # Coloca valores acima das barras
                for i, v in enumerate(df_bar["Quantidade"]):
                    ax_bar.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

                st.pyplot(fig_bar)

            # ------------------------------------------------------------------
            # Seção de distribuição de probabilidades (opcional)
            # ------------------------------------------------------------------
            if hasattr(model, "predict_proba"):
                st.subheader("Distribuição de Probabilidades de Acerto por Classe Real")
                # Cria DataFrame com apenas as features esperadas
                feature_cols = list(model.feature_names_in_)
                if "shot_made_flag" in feature_cols:
                    feature_cols.remove("shot_made_flag")

                df_features = df[feature_cols].copy()
                # Faz a previsão de probabilidade
                probas = model.predict_proba(df_features)[:, 1]
                df["proba"] = probas

                fig2, ax2 = plt.subplots(figsize=(10, 5))
                sns.histplot(
                    data=df,
                    x="proba",
                    hue="shot_made_flag",
                    bins=30,
                    kde=True,
                    palette={0: "salmon", 1: "skyblue"},
                    stat="count",
                    alpha=0.6,
                    multiple="layer",
                    common_norm=False,
                    ax=ax2
                )
                ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Limiar 0.5')
                ax2.set_xlim([0, 1])
                ax2.set_title("Distribuição de Probabilidades de Acerto por Classe Real", fontsize=14)
                ax2.set_xlabel("Probabilidade de Acerto Prevista")
                ax2.set_ylabel("Frequência")

                handles, labels = ax2.get_legend_handles_labels()
                new_labels = ["Erro (0)" if lab == "0" else "Acerto (1)" for lab in labels]
                ax2.legend(handles=handles[1:], labels=new_labels[1:], title="Classe Real", loc="upper right")

                st.pyplot(fig2)

    else:
        st.warning("⚠️ Arquivo de predições não encontrado. Execute o pipeline primeiro para visualizar os dados.")
