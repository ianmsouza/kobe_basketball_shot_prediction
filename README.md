# Projeto Final - Engenharia de Machine Learning (25E1_3)

## 🎯 Predição de Arremessos de Kobe Bryant com Machine Learning

Este projeto tem como objetivo desenvolver um modelo preditivo utilizando técnicas de Machine Learning para prever o sucesso dos arremessos realizados pelo famoso jogador de basquete Kobe Bryant durante sua carreira na NBA. O projeto explora abordagens de regressão logística e classificação com árvore de decisão para prever acertos ou erros dos arremessos.

## Estrutura do Projeto
O projeto segue o framework TDSP (Team Data Science Process) da Microsoft, organizado nas seguintes etapas principais:

- **Business Understanding:** Definir objetivos e requisitos do projeto.
- **Data Acquisition & Understanding:** Coleta e entendimento dos dados fornecidos.
- **Data Preparation:** Limpeza e processamento inicial dos dados.
- **Modelagem**: Desenvolvimento e avaliação dos modelos preditivos utilizando PyCaret, MLflow e Scikit-Learn.
- Regressão Logística
- Árvore de Decisão
- Avaliação usando Log Loss e F1 Score
- Registro e versionamento dos modelos
- **Deploy:** Implementação via MLflow como API local ou aplicação embarcada

## Artefatos Gerados
- Datasets processados
- Modelos treinados
- Scripts de processamento e treinamento
- Diagramas representando pipeline de dados e de aplicação
- Arquivos de configuração e requisitos (requirements.txt)

## Tecnologias Utilizadas
- Python 3.12.7
- PyCaret
- MLflow
- Scikit-Learn
- Pandas
- Streamlit (para visualização e monitoramento)

## Organização das Pastas
```
projeto_kobe_ml/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── diagrams/
│   └── reports/
├── src/
│   ├── data_processing.py
│   ├── treinamento.py
│   └── aplicacao.py
├── models/
├── notebooks/
├── scripts/
├── README.md
└── requirements.txt
```

## Como Executar
Para executar o projeto, crie um ambiente virtual conda com:
```bash
conda create -n infnet-25E1_3 python=3.12.7
conda activate infnet-25E1_3
pip install -r requirements.txt
```

## Link para os Dados
- [Dataset Kobe Bryant Shot Selection - Kaggle](https://www.kaggle.com/c/kobe-bryant-shot-selection/data)

## Monitoramento e Atualização do Modelo
- Monitoramento ativo e passivo com MLflow
- Estratégias de retrainamento reativa e preditiva
- Dashboard com Streamlit

## Autoria
- Ian Miranda de Souza

