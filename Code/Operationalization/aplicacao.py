"""
Módulo responsável pela aplicação do modelo treinado sobre dados de produção.

Este pipeline realiza:
- Carregamento do modelo final.
- Leitura da base de produção.
- Realização das predições com ajuste de threshold.
- Salvamento dos resultados com predições.
- Cálculo de métricas (Log Loss e F1 Score), se disponível a variável alvo.
- Registro das métricas e artefatos no MLflow com a rodada "PipelineAplicacao".
"""

import pandas as pd
from pycaret.classification import load_model
from sklearn.metrics import log_loss, f1_score
import mlflow
import logging
import os

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def aplicar_modelo(caminho_modelo, caminho_dados_producao, caminho_saida, threshold=0.35):
    """
    Executa a aplicação do modelo treinado sobre dados de produção.

    Esta função carrega um modelo treinado (PyCaret), realiza previsões sobre uma base de produção,
    salva os resultados e registra as métricas de desempenho no MLflow, caso a variável alvo esteja presente.

    Args:
        caminho_modelo (str): Caminho para o modelo salvo (sem extensão).
        caminho_dados_producao (str): Caminho para o arquivo .parquet com dados de produção.
        caminho_saida (str): Caminho do diretório para salvar os resultados com predições.
        threshold (float, opcional): Limite de probabilidade para converter predições em classe (default: 0.35).

    """
    logging.info("📦 Carregando modelo treinado...")
    modelo = load_model(caminho_modelo)

    logging.info("📥 Carregando dados de produção...")
    df_prod = pd.read_parquet(caminho_dados_producao)

    features = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance"]

    if not all(col in df_prod.columns for col in features):
        raise ValueError("❌ Dados de produção não contêm todas as features necessárias.")

    logging.info("🔮 Realizando predições com threshold ajustado...")

    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(df_prod[features])
        print("Shape das probabilidades:", probabilidades.shape)
        print("Primeiras linhas das probabilidades:")
        print(probabilidades[:5])
        print("Classes previstas pelo modelo:", modelo.classes_)

        # Aplica o threshold sobre a probabilidade da classe 1
        df_prod["prediction"] = (probabilidades[:, 1] >= threshold).astype(int)
    else:
        probabilidades = None
        df_prod["prediction"] = modelo.predict(df_prod[features])

    # Salvar os resultados
    os.makedirs(caminho_saida, exist_ok=True)
    output_path = os.path.join(caminho_saida, "predictions_prod.parquet")
    df_prod.to_parquet(output_path)
    logging.info(f"✅ Resultados salvos em {output_path}")

    # Definir experimento no MLflow
    mlflow.set_experiment("PipelineAplicacao")

    # Se a variável alvo estiver disponível, calcular métricas
    if "shot_made_flag" in df_prod.columns:
        valid_idx = df_prod["shot_made_flag"].notna()
        if valid_idx.sum() > 0:
            y_true = df_prod.loc[valid_idx, "shot_made_flag"]
            y_pred = df_prod.loc[valid_idx, "prediction"]

            metrics = {}
            if probabilidades is not None:
                prob_valid = probabilidades[valid_idx, 1]
                loss = log_loss(y_true, prob_valid, labels=[0, 1])
                metrics["log_loss_prod"] = loss

            f1 = f1_score(y_true, y_pred)
            metrics["f1_prod"] = f1

            logging.info(f"📊 Métricas calculadas: {metrics}")

            # Log da rodada no MLflow
            with mlflow.start_run(run_name="PipelineAplicacao"):
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(output_path)
            
                # Log da distribuição das predições
                pred_dist = df_prod["prediction"].value_counts(normalize=True).to_dict()
                mlflow.log_metrics({f"pred_class_{int(k)}": float(v) for k, v in pred_dist.items()})
        else:
            logging.warning("⚠️ Nenhuma linha com 'shot_made_flag' válida para avaliação.")
    else:
        logging.warning("⚠️ Coluna 'shot_made_flag' não está presente na base de produção.")

    # Estatísticas descritivas para debug e análise
    print(df_prod[features].describe())
    print("Distribuição do target:", df_prod["shot_made_flag"].value_counts())

if __name__ == "__main__":
    aplicar_modelo(
        caminho_modelo="../../Data/Modeling/modelo_final",
        caminho_dados_producao="../../Data/Raw/dataset_kobe_prod.parquet",
        caminho_saida="../../Data/Processed",
        threshold=0.35  # ajuste de limite de decisão
    )