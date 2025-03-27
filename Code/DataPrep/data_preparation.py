import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preparar_dados(caminho_base_dev, caminho_base_prod, caminho_saida):
    logging.info("🔍 Lendo os dados de desenvolvimento e produção...")
    df_dev = pd.read_parquet(caminho_base_dev)
    df_prod = pd.read_parquet(caminho_base_prod)

    # Selecionar colunas e remover NAs
    cols = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
    logging.info("🧹 Filtrando colunas e removendo valores nulos...")
    df_dev_filtered = df_dev[cols].dropna()
    df_prod_filtered = df_prod[cols].dropna()

    logging.info(f"✅ Dimensão do dataset filtrado (dev): {df_dev_filtered.shape}")

    # Criar diretório de saída se não existir
    os.makedirs(caminho_saida, exist_ok=True)

    # Salvar dataset filtrado
    df_dev_filtered.to_parquet(os.path.join(caminho_saida, "data_filtered.parquet"))

    # Separação treino/teste
    X = df_dev_filtered.drop('shot_made_flag', axis=1)
    y = df_dev_filtered['shot_made_flag']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Salvar splits
    logging.info("💾 Salvando bases de treino e teste...")
    X_train.join(y_train).to_parquet(os.path.join(caminho_saida, "base_train.parquet"))
    X_test.join(y_test).to_parquet(os.path.join(caminho_saida, "base_test.parquet"))

    # Log no MLflow
    logging.info("📊 Registrando parâmetros e métricas no MLflow...")

    # Define o experimento antes de iniciar o run
    mlflow.set_experiment("PreparacaoDados")

    with mlflow.start_run(run_name="PreparacaoDados"):
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("train_size", X_train.shape[0])
        mlflow.log_metric("test_size", X_test.shape[0])
        mlflow.log_metric("filtered_rows", df_dev_filtered.shape[0])

        # Contagem das classes
        class_counts = Counter(y)
        mlflow.log_metric("class_0_count", class_counts.get(0, 0))
        mlflow.log_metric("class_1_count", class_counts.get(1, 0))

    logging.info("✅ Pipeline de preparação de dados finalizado com sucesso.")

if __name__ == "__main__":
    preparar_dados(
        caminho_base_dev="../../Data/Raw/dataset_kobe_dev.parquet",
        caminho_base_prod="../../Data/Raw/dataset_kobe_prod.parquet",
        caminho_saida="../../Data/Processed"
    )
