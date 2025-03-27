import pandas as pd
from pycaret.classification import setup, create_model, pull, save_model
import mlflow
import logging
import os

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def treinar_modelos(caminho_base_treino, caminho_modelo_saida):
    logging.info("📆 Lendo base de treino...")
    df_train = pd.read_parquet(caminho_base_treino)

    # Define experimento no MLflow
    mlflow.set_experiment("Treinamento")

    # Inicia o run com nome explícito
    with mlflow.start_run(run_name="Treinamento"):
        logging.info("⚙️ Iniciando configuração do PyCaret...")
        setup(
            data=df_train,
            target='shot_made_flag',
            log_experiment=True,
            experiment_name='Treinamento',
            session_id=42
        )

        logging.info("🤖 Treinando modelo de Regressão Logística...")
        lr_model = create_model('lr')
        lr_metrics = pull()
        lr_f1 = lr_metrics.loc["Mean", "F1"]

        logging.info("🌳 Treinando modelo de Árvore de Decisão...")
        dt_model = create_model('dt')
        dt_metrics = pull()
        dt_f1 = dt_metrics.loc["Mean", "F1"]

        # Logar métricas manualmente
        mlflow.log_metric("f1_lr", lr_f1)
        mlflow.log_metric("f1_dt", dt_f1)

        logging.info(f"📊 F1 LR: {lr_f1:.4f} | F1 DT: {dt_f1:.4f}")
        final_model = dt_model if dt_f1 >= lr_f1 else lr_model

        os.makedirs(caminho_modelo_saida, exist_ok=True)
        caminho_completo = os.path.join(caminho_modelo_saida, "modelo_final")
        save_model(final_model, caminho_completo)
        logging.info(f"✅ Modelo final salvo em: {caminho_completo}.pkl")

if __name__ == "__main__":
    treinar_modelos(
        caminho_base_treino="../../Data/Processed/base_train.parquet",
        caminho_modelo_saida="../../Data/Modeling"
    )
