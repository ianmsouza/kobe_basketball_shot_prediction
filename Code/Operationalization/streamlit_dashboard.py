import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import mlflow
import tempfile

sns.set_style("whitegrid")

st.set_page_config(page_title="Dashboard Analítico - Modelo Kobe Bryant", layout="wide")
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

        # 📈 Comparativo de Arremessos do Kobe
        st.subheader("📈 Comparativo de Arremessos do Kobe")

        total_arremessos = len(df)
        acertos_previstos = int(df["prediction"].sum())
        erros_previstos = total_arremessos - acertos_previstos
        taxa_acerto = (acertos_previstos / total_arremessos) * 100

        col_esq, col_dir = st.columns([1, 3])

        with col_esq:
            st.markdown("<h5 style='color:white;'>Total de Arremessos</h5>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:white;'>{total_arremessos}</h2>", unsafe_allow_html=True)
            st.markdown("<h5 style='color:white;'>Acertos Previstos</h5>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:white;'>{acertos_previstos}</h2>", unsafe_allow_html=True)
            st.markdown("<h5 style='color:white;'>Taxa de Acerto (%)</h5>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:white;'>{taxa_acerto:.2f}%</h2>", unsafe_allow_html=True)

        with col_dir:
            fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
            sns.barplot(
                x=["Acertos Previstos", "Erros Previstos"],
                y=[acertos_previstos, erros_previstos],
                palette=["#1f77b4", "#aec7e8"],
                ax=ax_bar
            )
            ax_bar.set_ylabel("Quantidade")
            ax_bar.set_title("Acertos vs. Erros (Preditos)")
            for i, val in enumerate([acertos_previstos, erros_previstos]):
                ax_bar.text(i, val + 50, str(val), ha='center', fontweight='bold')
            st.pyplot(fig_bar)

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

            # Log no MLflow
            mlflow.set_experiment("PipelineAplicacao")
            with mlflow.start_run(run_name="Streamlit_Dashboard_Analitico"):
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

                acuracia = (y_true == y_pred).mean()
                f1_val = report["1"]["f1-score"] if "1" in report else 0.0
                recall_val = report["1"]["recall"] if "1" in report else 0.0
                precision_val = report["1"]["precision"] if "1" in report else 0.0

                mlflow.log_metric("accuracy", acuracia)
                mlflow.log_metric("f1_score", f1_val)
                mlflow.log_metric("recall", recall_val)
                mlflow.log_metric("precision", precision_val)
                mlflow.log_metric("acertos_previstos", acertos_previstos)
                mlflow.log_metric("taxa_acerto_percentual", taxa_acerto)

                # Salvar gráfico de acertos vs erros
                grafico_path = os.path.join(tempfile.gettempdir(), "grafico_acertos_vs_erros.png")
                fig_bar.savefig(grafico_path, bbox_inches="tight")
                mlflow.log_artifact(grafico_path, artifact_path="figuras")
                plt.close(fig_bar)

                # Salvar matriz de confusão como imagem
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Erro", "Acerto"],
                            yticklabels=["Erro", "Acerto"], ax=ax_cm)
                ax_cm.set_xlabel("Predito")
                ax_cm.set_ylabel("Real")
                ax_cm.set_title("Matriz de Confusão - Produção")
                cm_path = os.path.join(tempfile.gettempdir(), "confusion_matrix.png")
                fig_cm.savefig(cm_path, bbox_inches="tight")
                mlflow.log_artifact(cm_path, artifact_path="figuras")
                plt.close(fig_cm)

                # Salvar base de predições
                df_path = os.path.join(tempfile.gettempdir(), "df_predicoes.parquet")
                df.to_parquet(df_path, index=False)
                mlflow.log_artifact(df_path, artifact_path="dados")

                # Seção de distribuição de probabilidades
                if hasattr(model, "predict_proba"):
                    st.subheader("Distribuição de Probabilidades de Acerto por Classe Real")
                    feature_cols = list(model.feature_names_in_)
                    if "shot_made_flag" in feature_cols:
                        feature_cols.remove("shot_made_flag")

                    df_features = df[feature_cols].copy()
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

                    # Salvar histograma como artefato
                    probas_path = os.path.join(tempfile.gettempdir(), "distribuicao_probas.png")
                    fig2.savefig(probas_path, bbox_inches="tight")
                    mlflow.log_artifact(probas_path, artifact_path="figuras")
                    plt.close(fig2)

                st.success("📡 Execução registrada no MLflow com sucesso ✅")

    else:
        st.warning("⚠️ Arquivo de predições não encontrado. Execute o pipeline primeiro para visualizar os dados.")