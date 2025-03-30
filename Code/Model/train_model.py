import pandas as pd
import mlflow
import logging
import os

from pycaret.classification import setup, create_model, calibrate_model, finalize_model, save_model
from sklearn.metrics import log_loss, f1_score

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def treinar_modelos(caminho_treino, caminho_teste, caminho_saida):
    logging.info("📥 Carregando bases de treino e teste...")
    df_train = pd.read_parquet(caminho_treino)
    df_test = pd.read_parquet(caminho_teste)

    logging.info("⚙️ Configurando o ambiente do PyCaret...")
    s = setup(
        data=df_train,
        target="shot_made_flag",
        session_id=42,
        log_experiment=True,
        experiment_name="Treinamento",
        log_plots=False,
        verbose=False
    )

    modelos_info = {}

    for nome_modelo in ["lr", "dt"]:  # Regressão logística e árvore de decisão
        logging.info(f"🚀 Treinando modelo: {nome_modelo.upper()}")
        modelo = create_model(nome_modelo)
        modelo_calibrado = calibrate_model(modelo)
        modelo_final = finalize_model(modelo_calibrado)

        # Avaliação manual na base de teste
        X_test = df_test.drop(columns="shot_made_flag")
        y_true = df_test["shot_made_flag"]
        y_pred = modelo_final.predict(X_test)
        y_proba = modelo_final.predict_proba(X_test)[:, 1]

        loss = log_loss(y_true, y_proba, labels=[0, 1])
        f1 = f1_score(y_true, y_pred)

        modelos_info[nome_modelo] = {
            "modelo": modelo_final,
            "log_loss": loss,
            "f1_score": f1,
        }

        logging.info(f"📊 {nome_modelo.upper()} | Log Loss: {loss:.4f} | F1 Score: {f1:.4f}")

    # Selecionar melhor modelo com base no F1
    melhor_nome = max(modelos_info, key=lambda k: modelos_info[k]["f1_score"])
    melhor_modelo = modelos_info[melhor_nome]["modelo"]
    logging.info(f"✅ Modelo selecionado: {melhor_nome.upper()}")

    # Salvar modelo final
    os.makedirs(caminho_saida, exist_ok=True)
    caminho_modelo = os.path.join(caminho_saida, "modelo_final")
    save_model(melhor_modelo, caminho_modelo)
    logging.info(f"💾 Modelo salvo em: {caminho_modelo}.pkl")

    # Log no MLflow
    mlflow.set_experiment("Treinamento")
    mlflow.log_param("modelo_selecionado", melhor_nome)
    mlflow.log_metric("log_loss", modelos_info[melhor_nome]["log_loss"])
    mlflow.log_metric("f1_score", modelos_info[melhor_nome]["f1_score"])
    mlflow.log_artifact(f"{caminho_modelo}.pkl")

    logging.info("🏁 Pipeline de treinamento finalizado.")

if __name__ == "__main__":
    treinar_modelos(
        caminho_treino="../../Data/Processed/base_train.parquet",
        caminho_teste="../../Data/Processed/base_test.parquet",
        caminho_saida="../../Data/Modeling"
    )
