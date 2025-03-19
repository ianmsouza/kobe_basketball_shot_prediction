import streamlit as st
import pandas as pd
import mlflow

st.title("Monitoramento do Modelo")
st.write("## Métricas em Tempo Real")

# Buscar execuções ordenadas pela data de início (mais recente primeiro)
results = mlflow.search_runs(order_by=["attribute.start_time DESC"])

if results.empty:
    st.write("Nenhuma execução encontrada no MLflow.")
else:
    latest_run = results.iloc[0]
    # Obter as métricas, com fallback se não existirem
    log_loss_metric = latest_run.get("metrics.log_loss_prod", "N/A")
    f1_metric = latest_run.get("metrics.f1_prod", "N/A")
    
    st.metric("Log Loss (Produção)", log_loss_metric)
    st.metric("F1 Score (Produção)", f1_metric)

# Tente carregar os dados de predição
try:
    # Ajuste o caminho conforme sua estrutura de diretórios
    df_prod = pd.read_parquet("Data/Processed/predictions_prod.parquet")
    # Exibe um gráfico de barras com a contagem das predições
    st.bar_chart(df_prod['prediction'].value_counts())
except Exception as e:
    st.error(f"Erro ao carregar os dados de predição: {e}")
