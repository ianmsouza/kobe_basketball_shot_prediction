"""
Script principal para execução automatizada do pipeline completo do projeto.

Este pipeline executa, em sequência:
1. Preparação dos dados (DataPrep)
2. Treinamento do modelo (Model)
3. Aplicação em produção (Operationalization)
4. Orientação para execução dos dashboards com Streamlit

Cada etapa é executada via subprocesso chamando os scripts correspondentes.
"""

import subprocess

def executar_pipeline():
    """
    Executa o pipeline completo do projeto em quatro etapas:

    1. Preparação dos Dados: Executa o script de pré-processamento e divisão treino/teste.
    2. Treinamento do Modelo: Executa o script de treinamento com PyCaret e MLflow.
    3. Aplicação em Produção: Aplica o modelo treinado à base de produção.
    4. Dashboard: Exibe instruções para iniciar os dashboards interativos com Streamlit.

    Returns:
        None
    """
    print("\n=== Etapa 1: Preparação dos Dados ===")
    subprocess.run(["python", "../DataPrep/data_preparation.py"], check=True)

    print("\n=== Etapa 2: Treinamento do Modelo ===")
    subprocess.run(["python", "../Model/train_model.py"], check=True)

    print("\n=== Etapa 3: Aplicação em Produção ===")
    subprocess.run(["python", "aplicacao.py"], check=True)

    print("\n=== Etapa 4: Dashboard ===")
    print("\nInicie o dashboard com: streamlit run streamlit_dashboard.py")
    print("                        streamlit run streamlit_dashboard_mapa.py")
    print("                        streamlit run streamlit_dashboard_simulacao.py")

    print("\n✅ Pipeline executado com sucesso!")

if __name__ == "__main__":
    executar_pipeline()
