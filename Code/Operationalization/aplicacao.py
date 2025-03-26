import pandas as pd
from pycaret.classification import load_model
from sklearn.metrics import log_loss, f1_score
import mlflow
import logging
import os

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def aplicar_modelo(caminho_modelo, caminho_dados_producao, caminho_saida):
    logging.info("📦 Carregando modelo treinado...")
    modelo = load_model(caminho_modelo)

    logging.info("📥 Carregando dados de produção...")
    df_prod = pd.read_parquet(caminho_dados_producao)

    # Features usadas pelo modelo
    features = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance"]

    if not all(col in df_prod.columns for col in features):
        raise ValueError("❌ Dados de produção não contêm todas as features necessárias.")

    logging.info("🔮 Realizando predições...")
    predicoes = modelo.predict(df_prod[features])

    # Verifica se o modelo fornece probabilidades
    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(df_prod[features])
    else:
        probabilidades = None

    # Adiciona resultados ao DataFrame
    df_prod["prediction"] = predicoes

    os.makedirs(caminho_saida, exist_ok=True)
    output_path = os.path.join(caminho_saida, "predictions_prod.parquet")
    df_prod.to_parquet(output_path)
    logging.info(f"✅ Resultados salvos em {output_path}")

    # Registrar métricas se resposta estiver disponível
    if "shot_made_flag" in df_prod.columns:
        valid_idx = df_prod["shot_made_flag"].notna()
        if valid_idx.sum() > 0:
            y_true = df_prod.loc[valid_idx, "shot_made_flag"]
            y_pred = predicoes[valid_idx]

            metrics = {}
            if probabilidades is not None:
                prob_valid = probabilidades[valid_idx]
                loss = log_loss(y_true, prob_valid)
                metrics["log_loss_prod"] = loss

            f1 = f1_score(y_true, y_pred)
            metrics["f1_prod"] = f1

            logging.info(f"📊 Métricas calculadas: {metrics}")

            with mlflow.start_run(run_name="PipelineAplicacao"):
                mlflow.log_metrics(metrics)
        else:
            logging.warning("⚠️ Nenhuma linha com 'shot_made_flag' válida para avaliação.")
    else:
        logging.warning("⚠️ Coluna 'shot_made_flag' não está presente na base de produção.")

if __name__ == "__main__":
    aplicar_modelo(
        caminho_modelo="../../Data/Modeling/modelo_final",
        caminho_dados_producao="../../Data/Raw/dataset_kobe_prod.parquet",
        caminho_saida="../../Data/Processed"
    )
