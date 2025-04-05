"""
Módulo para treinamento dos modelos preditivos utilizando PyCaret e MLflow.

Este módulo realiza as seguintes etapas:
- Carregamento das bases de treino e teste.
- Configuração do ambiente do PyCaret para experimentos.
- Treinamento de dois modelos: Regressão Logística (lr) e Árvore de Decisão (dt).
- Calibração e finalização dos modelos.
- Avaliação dos modelos utilizando as métricas Log Loss e F1 Score.
- Seleção do melhor modelo com base no F1 Score.
- Salvamento do modelo final e registro dos parâmetros e métricas no MLflow.
"""

import pandas as pd
import mlflow
import logging
import os

from pycaret.classification import setup, create_model, calibrate_model, finalize_model, save_model
from sklearn.metrics import log_loss, f1_score

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def treinar_modelos(caminho_treino, caminho_teste, caminho_saida):
    """
    Executa o pipeline de treinamento dos modelos utilizando PyCaret.

    As etapas do processo são:
    - Carregar as bases de treino e teste.
    - Configurar o ambiente do PyCaret com a base de treino.
    - Treinar os modelos de Regressão Logística e Árvore de Decisão.
    - Calibrar e finalizar cada modelo.
    - Avaliar os modelos na base de teste (cálculo de Log Loss e F1 Score).
    - Selecionar o melhor modelo com base no F1 Score.
    - Salvar o modelo final e registrar os artefatos e métricas no MLflow.

    Args:
        caminho_treino (str): Caminho para o arquivo .parquet com a base de treino.
        caminho_teste (str): Caminho para o arquivo .parquet com a base de teste.
        caminho_saida (str): Caminho do diretório para salvar o modelo final.

    Returns:
        None
    """
    logging.info("📥 Carregando bases de treino e teste...")
    df_train = pd.read_parquet(caminho_treino)
    df_test = pd.read_parquet(caminho_teste)

    # ⚠️ IMPORTANTE:
    # Embora o PyCaret permita configurar o tipo de validação cruzada via o parâmetro `fold_strategy`,
    # versões atuais (como a 3.3.2) não aceitam o valor 'stratifiedkfold' diretamente neste parâmetro.
    # No entanto, como a variável alvo `shot_made_flag` é binária (0 ou 1),
    # o PyCaret automaticamente aplica `StratifiedKFold` como estratégia de validação cruzada.
    # Por isso, não é necessário (nem possível) configurar isso manualmente — já está garantido internamente.
    # A configuração `fold=10` abaixo define explicitamente o número de dobras da validação cruzada.

    logging.info("⚙️ Configurando o ambiente do PyCaret...")
    s = setup(
        data=df_train,
        target="shot_made_flag",
        session_id=42,
        log_experiment=True,
        experiment_name="Treinamento",
        log_plots=False,
        fold=10,
        verbose=False
    )

    modelos_info = {}

    # Loop para treinar os modelos "lr" (regressão logística) e "dt" (árvore de decisão)
    for nome_modelo in ["lr", "dt"]:
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

    # Selecionar o melhor modelo com base no F1 Score
    melhor_nome = max(modelos_info, key=lambda k: modelos_info[k]["f1_score"])
    melhor_modelo = modelos_info[melhor_nome]["modelo"]
    logging.info(f"✅ Modelo selecionado: {melhor_nome.upper()}")

    # Salvar o modelo final
    os.makedirs(caminho_saida, exist_ok=True)
    caminho_modelo = os.path.join(caminho_saida, "modelo_final")
    save_model(melhor_modelo, caminho_modelo)
    logging.info(f"💾 Modelo salvo em: {caminho_modelo}.pkl")

    # Registro dos parâmetros e métricas no MLflow
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
