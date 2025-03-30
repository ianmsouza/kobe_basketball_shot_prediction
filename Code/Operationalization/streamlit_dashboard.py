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

# Configuração da página
st.set_page_config(page_title="Dashboard - Modelo Kobe Bryant", layout="wide")
st.title("📊 Dashboard Analítico - Modelo Kobe Bryant")

# Caminhos para arquivos
caminho_predicoes = "../../Data/Processed/predictions_prod.parquet"
caminho_modelo = "../../Data/Modeling/modelo_final"  # Corrigido

# Carregamento do modelo
model = load_model(caminho_modelo)

# Layout lateral com instruções
st.sidebar.title("📘 Sobre este Dashboard")
st.sidebar.markdown("""
Visualização e avaliação do desempenho do modelo de classificação de arremessos do Kobe Bryant.

**Como usar:**
- ✅ Execute o pipeline de aplicação
- 📊 Este painel analisará a produção atual
- 🏀 Para simulação de novas jogadas, use o arquivo:

```bash
streamlit run Code/Operationalization/streamlit_dashboard_simulacao.py
```
""")

# Avaliação analítica
st.markdown("""
### Avaliação Analítica do Modelo
O modelo foi treinado com 16.228 registros e avaliado com a base de produção. Esta interface permite:

- Ver amostras reais com predições
- Avaliar métricas como Accuracy, F1, Recall, LogLoss
- Visualizar a matriz de confusão
- Explorar a distribuição de probabilidades previstas
""")

# Verifica existência do arquivo
if os.path.exists(caminho_predicoes):
    df = pd.read_parquet(caminho_predicoes)

    st.subheader("▶️ Amostra das predições da produção")
    st.dataframe(df.head(10))

    if "shot_made_flag" in df.columns:
        y_true = df["shot_made_flag"].dropna()
        y_pred = df.loc[y_true.index, "prediction"]

        # Relatório de classificação
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()

        # Comparativo de arremessos com colunas
        st.subheader("📈 Comparativo de Arremessos do Kobe")
        col_esq, col_dir = st.columns([1, 3])

        with col_esq:
            total = len(df)
            acertos = sum(df["prediction"] == 1)
            erros = total - acertos
            taxa_acerto = 100 * acertos / total

            st.markdown("**Total de Arremessos**")
            st.markdown(f"<h2 style='color:white'>{total}</h2>", unsafe_allow_html=True)

            st.markdown("**Acertos Previstos**")
            st.markdown(f"<h2 style='color:white'>{acertos}</h2>", unsafe_allow_html=True)

            st.markdown("**Taxa de Acerto (%)**")
            st.markdown(f"<h2 style='color:white'>{taxa_acerto:.2f}%</h2>", unsafe_allow_html=True)

        with col_dir:
            fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
            sns.barplot(x=["Acertos Previstos", "Erros Previstos"],
                        y=[acertos, erros],
                        palette=["#1f77b4", "#aec7e8"],
                        ax=ax_bar)
            for i, val in enumerate([acertos, erros]):
                ax_bar.text(i, val + 100, str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax_bar.set_ylabel("Quantidade")
            ax_bar.set_xlabel("")
            ax_bar.set_title("Acertos vs. Erros (Preditos)")
            st.pyplot(fig_bar)

        # Métricas
        st.subheader("📊 Avaliação do Modelo")
        st.dataframe(report_df.style.format("{:.2f}"))

        # Matriz de confusão
        st.subheader("🔢 Matriz de Confusão")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Erro", "Acerto"],
                    yticklabels=["Erro", "Acerto"], ax=ax_cm)
        ax_cm.set_xlabel("Predito")
        ax_cm.set_ylabel("Real")
        st.pyplot(fig_cm)

        # Registro no MLflow
        mlflow.set_experiment("PipelineAplicacao")
        with mlflow.start_run(run_name="StreamlitDashboardAnalitico"):
            f1_score = report.get("1", {}).get("f1-score", 0.0)
            recall = report.get("1", {}).get("recall", 0.0)
            precision = report.get("1", {}).get("precision", 0.0)

            mlflow.log_metrics({
                "accuracy": (y_true == y_pred).mean(),
                "f1_score": f1_score,
                "recall": recall,
                "precision": precision
            })

            cm_path = os.path.join(tempfile.gettempdir(), "confusion_matrix.png")
            fig_cm.savefig(cm_path, bbox_inches="tight")
            mlflow.log_artifact(cm_path, artifact_path="figuras")
            plt.close(fig_cm)

            df_path = os.path.join(tempfile.gettempdir(), "df_predicoes.parquet")
            df.to_parquet(df_path, index=False)
            mlflow.log_artifact(df_path, artifact_path="dados")

            # Distribuição de probabilidades
            if hasattr(model, "predict_proba"):
                st.subheader("Distribuição de Probabilidades de Acerto por Classe Real")
                feature_cols = list(model.feature_names_in_)
                if "shot_made_flag" in feature_cols:
                    feature_cols.remove("shot_made_flag")

                probas = model.predict_proba(df[feature_cols])[:, 1]
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
                ax2.legend(title="Classe Real", loc="upper right")

                st.pyplot(fig2)

                probas_path = os.path.join(tempfile.gettempdir(), "distribuicao_probas.png")
                fig2.savefig(probas_path, bbox_inches="tight")
                mlflow.log_artifact(probas_path, artifact_path="figuras")
                plt.close(fig2)

        st.success("📡 Execução registrada no MLflow com sucesso ✅")
else:
    st.warning("⚠️ Arquivo de predições não encontrado. Execute o pipeline primeiro para visualizar os dados.")