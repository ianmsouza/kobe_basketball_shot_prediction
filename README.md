# Projeto Final - Engenharia de Machine Learning (25E1_3)

## 🎯 Predição de Arremessos de Kobe Bryant com Machine Learning

Este projeto tem como objetivo desenvolver um modelo preditivo utilizando técnicas de Machine Learning para prever o sucesso dos arremessos realizados pelo famoso jogador de basquete Kobe Bryant durante sua carreira na NBA. O projeto explora abordagens de regressão logística e classificação com árvore de decisão para prever acertos ou erros dos arremessos.

---

## 🧱 Estrutura do Projeto (TDSP)

```
infnet-25E1_3/
├── Code/
│   ├── DataPrep/
│   │   └── data_preparation.py
│   ├── Model/
│   │   └── train_model.py
│   └── Operationalization/
│       ├── mlruns/
│       ├── logs.log
│       ├── aplicacao.py
│       ├── main_pipeline.py
│       ├── streamlit_dashboard_mapa.py
│       ├── streamlit_dashboard_simulacao.py
│       └── streamlit_dashboard.py
├── Data/
│   ├── Logs/
│   │   └── simulacoes.csv
│   ├── Raw/
│   │   ├── dataset_kobe_dev.parquet
│   │   └── dataset_kobe_prod.parquet
│   ├── Processed/
│   │   ├── data_filtered.parquet
│   │   ├── base_train.parquet
│   │   ├── base_test.parquet
│   │   └── predictions_prod.parquet
│   ├── Modeling/
│   │   ├── modelo_final.pkl
├── Docs/
│   ├── Imagens/
│   │   └── charlotte_key_zone.jpeg
│   ├── Diagramas/
│   │   └── fluxograma_questao2.png
│   ├── Project/
│   │   └── execucao_pipeline.md
├── Notebooks/
│   ├── analisys_roc.ipynb
│   ├── criar_diagrama_questao_2.ipynb
│   └── criar_estrutura_tdsp.ipynb
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 🚀 Como Executar

### 1. Criar ambiente Conda e instalar dependências:
```bash
conda create -n infnet-25E1_3_v2 python=3.9.21
conda activate infnet-25E1_3_v2
pip install -r requirements.txt
```
ou YAML 
```bash
conda env create -f environment.yml
```

### 2. Rodar o pipeline completo:
```bash
# Executar todas as etapas em sequência
python Code/Operationalization/main_pipeline.py
```

### 3. Rodar o dashboard:
```bash
# Dashboard analítico com métricas e gráficos
streamlit run Code/Operationalization/streamlit_dashboard.py

# Mapa interativo dos arremessos
streamlit run Code/Operationalization/streamlit_dashboard_mapa.py

# Simulador de jogadas
streamlit run Code/Operationalization/streamlit_dashboard_simulacao.py
```

### 4. Ver o MLflow (opcional):
```bash
mlflow ui
```
Acesse: [http://localhost:5000](http://localhost:5000)

---

## 🛠️ Tecnologias Utilizadas
- Python 3.9.21
- PyCaret 3.3.2
- Scikit-Learn 1.4.2
- MLflow 1.30.0
- Streamlit 1.43.2
- Pandas 1.5.3
- NumPy 1.26.4
- Matplotlib 3.7.5
- Seaborn 0.13.2

---

## 📈 Métricas Finais (Produção)

| **Métrica**           | **Valor**         |
|-----------------------|-------------------|
| Modelo Escolhido      | Regressão Logística |
| Log Loss (Produção)   | 0.6289            |
| F1-Score (Produção)   | 0.1645             |

> 🔍 O modelo de **Regressão Logística** foi selecionado para produção por apresentar **desempenho mais consistente e estabilidade no ambiente de produção**, mesmo com F1-Score modesto.
>
> 📉 A **Árvore de Decisão** apresentou overfitting (F1-Score de 0.1072 em produção), enquanto a regressão logística manteve uma generalização mais robusta.

---

## 📊 Dataset
- [Kaggle - Kobe Bryant Shot Selection](https://www.kaggle.com/c/kobe-bryant-shot-selection/data)
- [Dados de desenvolvimento e produção](https://github.com/tciodaro/eng_ml/tree/main/data)

---

## 🧠 Projeto para a disciplina:
**Engenharia de Machine Learning (25E1_3)**  
**Instituto Infnet – 2025**

---

## 📝 Observações finais
- Todos os experimentos e métricas são registrados no MLflow
- O pipeline automatizado contempla 3 etapas: preparação → treinamento → aplicação
- Cada etapa registra uma rodada específica no MLflow:
  - `PreparacaoDados`
  - `Treinamento`
  - `PipelineAplicacao`

<br>

# **Respostas do projeto**

### **Questão 1)**
#### A solução criada nesse projeto deve ser disponibilizada em repositório git e disponibilizada em servidor de repositórios (Github (recomendado), Bitbucket ou Gitlab). O projeto deve obedecer o Framework TDSP da Microsoft (estrutura de arquivos, arquivo requirements.txt e arquivo README - com as respostas pedidas nesse projeto, além de outras informações pertinentes). Todos os artefatos produzidos deverão conter informações referentes a esse projeto (não serão aceitos documentos vazios ou fora de contexto). Escreva o link para seu repositório. 

>Resposta:
> <br>
> Link do GitHub: [https://github.com/ianmsouza/kobe_basketball_shot_prediction](https://github.com/ianmsouza/kobe_basketball_shot_prediction)

### **Questão 2)**
#### Iremos desenvolver um preditor de arremessos usando duas abordagens (regressão e classificação) para prever se o "Black Mamba" (apelido de Kobe) acertou ou errou a cesta.<br><br>Baixe os dados de desenvolvimento e produção [aqui](https://github.com/tciodaro/eng_ml/tree/main/data) (datasets: dataset_kobe_dev.parquet e dataset_kobe_prod.parquet). Salve-os numa pasta /data/raw na raiz do seu repositório.<br><br>Para começar o desenvolvimento, desenhe um diagrama que demonstra todas as etapas necessárias para esse projeto, desde a aquisição de dados, passando pela criação dos modelos, indo até a operação do modelo.

> Resposta:
> <br><br>
> **1️⃣ Aquisição de Dados**
> - Coleta dos dados brutos fornecidos (`dataset_kobe_dev.parquet` e `dataset_kobe_prod.parquet`).
> - Armazenamento na pasta `/Data/Raw`.
>
> **2️⃣ Pré-processamento dos Dados**
> - Remoção de valores ausentes.
> - **Aplicação de clipping** em features para limitar outliers (ex: `shot_distance` entre 0 e 35 pés).
> - Seleção das colunas relevantes: `lat`, `lon`, `minutes_remaining`, etc.
> - Salvamento dos dados tratados em `/Data/Processed`.
>
> ℹ️ *Nota:* Os limites de clipping foram definidos com base na distribuição histórica dos dados de treino e em conhecimento de domínio:
> 
> - `shot_distance` (0-35 pés): Arremessos além de 35 pés são raros na NBA e geralmente pouco precisos.
> - `lat` (33.2 - 34.1) e `lon` (-118.52 - -118.02): Coordenadas geográficas da quadra do Staples Center em Los Angeles, onde Kobe jogou a maior parte da carreira.
>
> **3️⃣ Separação em Treino/Teste**
> - Separação estratificada dos dados (80% treino, 20% teste).
> - Bases armazenadas em `/Data/Processed/base_train.parquet` e `base_test.parquet`.
>
> **4️⃣ Treinamento dos Modelos**
> - Modelos: **Regressão Logística e Árvore de Decisão**.
> - Ferramentas: **PyCaret, MLFlow** para rastreamento de experimentos.
>
> **5️⃣ Avaliação dos Modelos**
> - Cálculo de métricas: **Log-Loss e F1-score**.
> - Comparação dos modelos para seleção do melhor.
>
> **6️⃣ Deploy/Operacionalização**
> - O modelo escolhido é armazenado e carregado para previsões em produção.
> - Implementação realizada com **MLflow** para versionamento e rastreamento do modelo, e **Streamlit** para dashboards interativos.
>
> **7️⃣ Monitoramento do Modelo**
> - Monitoramento contínuo da performance do modelo em produção com registro de métricas como **Log Loss** e **F1 Score** no **MLflow**.
> - Análises visuais e interativas com **Streamlit**, permitindo acompanhamento da saúde do modelo, comparação entre acertos e erros, e detecção de possíveis desvios de comportamento (drift).
>
> **8️⃣ Atualização do Modelo**
> - Estratégias:
>   - **Reativa**: Atualiza quando a performance do modelo cai.
>   - **Preditiva**: Prevê mudanças e ajusta o modelo antes da degradação.
> 
> **Diagrama**
> <br><br>
> ![Diagrama](/Docs/Diagramas/fluxograma_questao2.png)
> <br><br>
> ![Diagrama](/Docs/Diagramas/Diagrama_do_projeto.drawio.png)
>

### **Questão 3)**
#### Como as ferramentas Streamlit, MLFlow, PyCaret e Scikit-Learn auxiliam na construção dos pipelines descritos anteriormente? A resposta deve abranger os seguintes aspectos: <br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a. Rastreamento de experimentos;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b. Funções de treinamento;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;c. Monitoramento da saúde do modelo;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;d. Atualização de modelo;<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;e. Provisionamento (Deployment).
> Resposta:
><br><br>
> A construção do pipeline de Machine Learning segue a estrutura definida no Framework TDSP, permitindo que cada ferramenta desempenhe um papel específico dentro do fluxo de trabalho.
> <br>
> A seguir, uma explicação de como Streamlit, MLFlow, PyCaret e Scikit-Learn auxiliam em cada uma das fases do projeto:
> <br><br>
> **a. Rastreamento de Experimentos**
> <br>
> 🛠️ Ferramenta principal: MLFlow
> - O MLFlow permite registrar métricas, parâmetros e artefatos de cada experimento.
> - Isso facilita a comparação de diferentes modelos e versões treinadas.
> - No nosso pipeline, cada modelo será registrado no MLFlow, garantindo rastreabilidade.
>
> 🛠️ PyCaret também auxilia
> <br>
> - O PyCaret já possui integração nativa com o MLFlow, facilitando o log automático dos experimentos.
> - Isso simplifica o rastreamento sem precisar adicionar código extra.
> 
> **b. Funções de Treinamento**
> <br>🛠️ Ferramentas principais: PyCaret e Scikit-Learn
> - PyCaret automatiza o processo de treinamento, permitindo testar múltiplos modelos rapidamente.
> - Scikit-Learn fornece as bibliotecas base para treinar modelos, como Regressão Logística e Árvore de Decisão.
> - No nosso pipeline, utilizamos PyCaret para selecionar e avaliar os melhores modelos.
> 
> - O MLFlow pode ser usado para registrar os modelos treinados, salvando artefatos para deploy futuro.
>
> **c. Monitoramento da Saúde do Modelo**
> <br>🛠️ Ferramentas principais: MLFlow e Streamlit
> - MLFlow armazena logs das execuções dos modelos em produção, possibilitando a análise de degradação do desempenho.
> - Streamlit pode ser usado para criar dashboards interativos e monitorar métricas como Log Loss e F1-score.
> 
> **d. Atualização do Modelo**
> <br>🛠️ Ferramentas principais: MLFlow e PyCaret
> - O modelo pode ser atualizado através de estratégias reativas e preditivas.
> - O MLFlow permite versionar diferentes treinamentos, facilitando a troca do modelo sempre que houver degradação.
> - O PyCaret facilita o re-treinamento do modelo de forma simples:
> - Essa abordagem facilita a implantação de novos modelos sem impactar a operação.
>
> **e. Provisionamento (Deployment)**
> <br>🛠️ Ferramentas principais: MLFlow e Streamlit
> - O MLFlow Models permite exportar e servir modelos automaticamente como uma API:
> - Streamlit pode ser usado para criar uma interface gráfica, permitindo que usuários façam previsões diretamente pelo navegador.
>
> Isso facilita a interação com o modelo sem precisar de habilidades técnicas.
>
> **Conclusão Final**
>
> Cada ferramenta desempenha um papel fundamental no pipeline de Machine Learning:
>
| **Ferramenta**    | **Função Principal** |
|-------------------|-----------------------------------------------|
| **MLFlow**       | Rastreamento de experimentos, versionamento de modelos e monitoramento |
| **PyCaret**      | Automação de treinamento e comparação de modelos |
| **Scikit-Learn** | Implementação dos modelos clássicos de Machine Learning |
| **Streamlit**    | Construção de dashboards para visualização e deploy interativo |
>
> Com essa abordagem, garantimos um pipeline eficiente, rastreável e totalmente operacional, atendendo às exigências do TDSP e permitindo um fluxo contínuo de treinamento, avaliação, deploy e monitoramento. 

### **Questão 4)**
#### Com base no diagrama realizado na questão 2, aponte os artefatos que serão criados ao longo de um projeto. Para cada artefato, a descrição detalhada de sua composição.

> Resposta:

| Artefato | Descrição |
|----------|-----------|
| Data/Processed/data_filtered.parquet | Conjunto de dados filtrado com colunas relevantes e sem valores ausentes, utilizado como base para modelagem. |
| Data/Processed/base_train.parquet | Subconjunto de dados estratificado (80%) utilizado para o treinamento dos modelos. |
| Data/Processed/base_test.parquet | Subconjunto de dados estratificado (20%) utilizado para avaliação da performance dos modelos. |
| Data/Processed/predictions_prod.parquet | Arquivo contendo as predições geradas pelo modelo final aplicadas à base de produção. |
| Data/Modeling/modelo_final.pkl | Modelo final treinado e serializado com PyCaret, pronto para ser servido em ambiente produtivo. |
| Data/Logs/simulacoes.csv | Histórico das simulações realizadas no dashboard interativo de simulação desenvolvido com Streamlit. |
| Code/DataPrep/preparacao_dados.py | Script responsável pela preparação e limpeza dos dados brutos, incluindo filtragem de colunas e remoção de nulos. |
| Code/Model/train_model.py | Notebook contendo o pipeline de treinamento dos modelos, registro no MLflow e avaliação de métricas. |
| Code/Operationalization/aplicacao.py | Script para operacionalização do modelo via API local, permitindo inferência externa. |
| Code/Operationalization/streamlit_dashboard.py | Dashboard desenvolvido em Streamlit para visualização de métricas e monitoramento do modelo em produção. |
| Code/Operationalization/streamlit_dashboard_simulacao.py | Dashboard desenvolvido em Streamlit para simulações e mapa de Arremesso. |
| Code/Operationalization/streamlit_dashboard_mapa.py | Dashboard desenvolvido em Streamlit da localização dos arremessos - Kobe Bryant. |
|Docs/Diagramas/fluxograma_questao2.png	| Diagrama representando as etapas do pipeline segundo o TDSP |
|Docs/Diagramas/Diagrama_do_projeto.drawio.png | Arquitetura detalhada do fluxo de dados e execução do projeto |

### **Questão 5)**
#### Implemente o pipeline de processamento de dados com o mlflow, rodada (run) com o nome "PreparacaoDados":<br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a. Os dados devem estar localizados em "/data/raw/dataset_kobe_dev.parquet" e "/data/raw/dataset_kobe_prod.parquet"<br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b.Observe que há dados faltantes na base de dados! As linhas que possuem dados faltantes devem ser desconsideradas. Para esse exercício serão apenas consideradas as colunas:<br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i. lat<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ii. lng<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;iii. minutes remaining<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;iv. period<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v. playoffs<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;vi. shot_distance<br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A variável shot_made_flag será seu alvo, onde 0 indica que Kobe errou e 1 que a cesta foi realizada. O dataset resultante será armazenado na pasta "/data/processed/data_filtered.parquet". Ainda sobre essa seleção, qual a dimensão resultante do dataset?<br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;vii. Separe os dados em treino (80%) e teste (20 %) usando uma escolha aleatória e estratificada. Armazene os datasets resultantes em "/Data/processed/base_{train|test}.parquet . Explique como a escolha de treino e teste afetam o resultado do modelo final. Quais estratégias ajudam a minimizar os efeitos de viés de dados.<br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; viii. Registre os parâmetros (% teste) e métricas (tamanho de cada base) no MlFlow

> Resposta:
>
> a.<br>
> Os dados utilizados foram carregados a partir dos arquivos:
> - `/Data/Raw/dataset_kobe_dev.parquet` (base de desenvolvimento)
> - `/Data/Raw/dataset_kobe_prod.parquet` (base de produção, usada posteriormente na aplicação)
> 
> b.<br>
> Durante a etapa de preparação, foram selecionadas apenas as seguintes colunas, conforme solicitado:
> 
> - `lat`: latitude 
> - `lon`: longitude 
> - `minutes_remaining`: minutos restantes no período da partida.
> - `period`: número do período (1 a 4 ou prorrogações).
> - `playoffs`: indica se a partida é de playoff (1) ou temporada regular (0).
> - `shot_distance`: distância do arremesso até a cesta (em pés).
> - `shot_made_flag`: variável alvo. Valor 1 indica acerto, valor 0 indica erro.
> 
> Linhas com qualquer valor nulo nessas colunas foram removidas, como etapa de limpeza obrigatória. Após essa filtragem, o dataset foi salvo em: `/Data/Processed/data_filtered.parquet`
>
> A dimensão resultante do dataset após o filtro foi:
> - **20.285 linhas**
> - **7 colunas**
>
> As 7 colunas finais foram: `lat`, `lon`, `minutes_remaining`, `period`, `playoffs`, `shot_distance`, `shot_made_flag`.
>
> Essa versão processada dos dados foi registrada no MLflow na rodada chamada `"PreparacaoDados"`, junto com os parâmetros e métricas utilizadas.
> 
> Separação entre treino e teste com amostragem estratificada<br>
> Após a filtragem, os dados foram divididos em:
> - **80% para treino**
> - **20% para teste**
>
> Utilizamos a técnica de **amostragem estratificada** com base na variável `shot_made_flag`, o que garante que a proporção entre acertos e erros seja mantida em ambos os conjuntos.
>
> Os arquivos gerados foram:
> - `/Data/Processed/base_train.parquet`
> - `/Data/Processed/base_test.parquet`
>
> Essa divisão foi essencial para que o modelo fosse avaliado de maneira justa em dados **não vistos**, simulando um ambiente de produção real.
>
> A separação estratificada ajuda a manter a proporção de classes entre treino e teste. Para minimizar viés, é importante garantir que o conjunto de treino seja representativo do domínio do problema, usar validação cruzada e monitorar métricas de performance em dados fora da amostra.
>
> 📊 Registro no MLflow:
> 
> Durante a etapa `"PreparacaoDados"`, registramos os seguintes parâmetros e métricas no MLflow:
>- **Parâmetro**
>  <br>`test_size`: 0.2
>
>- **Métricas**
>  <br>`Total filtrado (dados limpos)`: 20.285 linhas e 7 colunas
>  <br>`Base de treino`: 16.228 linhas (80%)
>  <br>`Base de teste`: 4.057 linhas (20%)
>  <br>`Train (PyCaret após split interno)`: 11.359 linhas
>  <br>`Test (PyCaret após split interno)`: 4.869 linhas
> 
> Essas informações são importantes para rastreabilidade do experimento e ajudam na reprodutibilidade do pipeline ao longo do tempo.
> 
### **Questão 6)**
#### Implementar o pipeline de treinamento do modelo com o MlFlow usando o nome "Treinamento"<br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a.Com os dados separados para treinamento, treine um modelo com regressão logística do sklearn usando a biblioteca pyCaret.<br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b.Registre a função custo "log loss" usando a base de teste<br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;c.Com os dados separados para treinamento, treine um modelo de árvore de decisão do sklearn usando a biblioteca pyCaret.<br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;d.Registre a função custo "log loss" e F1_score para o modelo de árvore.<br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;e.Selecione um dos dois modelos para finalização e justifique sua escolha.

> Resposta:
> 
> A partir das bases geradas no pipeline de preparação (`base_train.parquet` e `base_test.parquet`), realizamos o treinamento de dois modelos de classificação usando a biblioteca PyCaret e rastreamento com MLflow, na rodada `"Treinamento"`.
> <br><br>
> **a. Regressão Logística com PyCaret**
> 
> O primeiro modelo treinado foi uma **regressão logística**, utilizando a função `create_model("lr")` do PyCaret, com os dados de treino.
> 
> O PyCaret foi configurado com:
> - Target: `shot_made_flag`
> - Fold cross-validation: `StratifiedKFold`, com 10 dobras
> - Logging no MLflow: ativado com `log_experiment=True`
> 
> **b. Registro da função de custo (log loss) – Regressão Logística**
> 
> O desempenho do modelo de regressão logística foi avaliado com base na métrica **log loss** utilizando os dados de teste.  
> Essa métrica foi registrada no MLflow com a tag `"log_loss_lr"`.
> 
> 📊 **Resultados da Regressão Logística:**
> - **Log Loss (teste):** `0.6785`
> - **F1 Score (teste):** `0.5129`
> - **F1 Score (cross-val):** `0.5240`
> 
> 📦 **Desempenho em Produção:**
> - **Log Loss (produção):** `0.6289`
> - **F1 Score (produção):** `0.1645`
> 
> **c. Árvore de Decisão com PyCaret**
> 
> Em seguida, treinamos um modelo de **árvore de decisão**, com a função `create_model("dt")` do PyCaret, utilizando os mesmos dados.
> 
> **d. Registro das métricas – Árvore de Decisão**
> 
> Para o modelo de árvore de decisão, foram registradas as seguintes métricas no MLflow:
> - **Log Loss (teste)**: `0.6903`
> - **F1 Score (teste)**: `0.1072`
> - **F1 Score (cross-validation)**: `0.5392`
> 
> As métricas foram registradas com as tags:
> - `"log_loss_dt"`
> - `"f1_dt"`
> 
> **e. Escolha do modelo final**
> 
> O modelo selecionado para uso em produção foi a **Regressão Logística**, pelos seguintes motivos:
> 
> - Apresentou **menor Log Loss em teste** (`0.6785`) e **em produção** (`0.6289`);
> - Teve **comportamento mais estável** entre treino, teste e produção;
> - A árvore de decisão apresentou sinais de **overfitting**, com desempenho razoável no treino mas queda significativa nos dados reais;
> - A regressão logística é um modelo **mais robusto e generalizável**, o que é desejável para operação contínua.
> 
> O modelo final foi salvo como `modelo_final.pkl` na pasta `/data/modeling/`, e a rodada de treinamento foi registrada no MLflow com o nome `"Treinamento"`.
>
>
> **📈 Comparativo Final das Métricas (Produção)**
>
| Modelo              | Log Loss | F1 Score |
|---------------------|----------|----------|
| Regressão Logística | 0.62888  | 0.1645   |
| Árvore de Decisão   | 0.6903   | 0.1072   |

![Pipeline Status](https://img.shields.io/badge/pipeline-success-brightgreen)


### **Questão 7)**
#### Registre o modelo de classificação e o sirva através do MLFlow (ou como uma API local, ou embarcando o modelo na aplicação). Desenvolva um pipeline de aplicação (aplicacao.py) para carregar a base de produção (/data/raw/dataset_kobe_prod.parquet) e aplicar o modelo. Nomeie a rodada (run) do mlflow como “PipelineAplicacao” e publique, tanto uma tabela com os resultados obtidos (artefato como .parquet), quanto log as métricas do novo log loss e f1_score do modelo.<br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a.O modelo é aderente a essa nova base? O que mudou entre uma base e outra? Justifique.<br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b.Descreva como podemos monitorar a saúde do modelo no cenário com e sem a disponibilidade da variável resposta para o modelo em operação.<br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;c.Descreva as estratégias reativa e preditiva de retreinamento para o modelo em operação.

> Resposta:
>
> **Implementação:**
>
> - O modelo foi servido via `aplicacao.py`, que carrega a base de produção, aplica predições com **threshold=0.35** e registra métricas no MLflow.
>
> ℹ️ *Nota:* O threshold de 0.35 foi definido após análise da curva ROC e otimização do F1-Score na base de validação. Esse valor equilibra a taxa de acertos (recall) e a precisão, garantindo que o modelo não seja excessivamente conservador nem gere muitos falsos positivos.
>
> **a. Aderência do Modelo:**
> 
> - **Aderência parcial:** A base de produção apresenta diferenças na distribuição de `shot_distance` e `period`, resultando em F1-Score menor (0.1645 vs. 0.5129 em teste).
> 
> **b. Monitoramento:**
> 
> - **Com variável resposta:** Acompanhamento de Log Loss e F1-Score.
> 
> - **Sem variável resposta:** Análise de distribuição de features (PSI, KS Test) e confiança das predições.
> 
> **c. Estratégias de Retreinamento:**
> 
> - **Reativa:** Reentrena o modelo após detecção de queda no F1-Score.
> 
> - **Preditiva:** Reentrena com base em alertas de drift (ex: mudança na distribuição de `shot_distance`).
> 

### **Questão 8)**
#### Implemente um dashboard de monitoramento da operação usando Streamlit.

> Resposta:
>
> Para viabilizar o monitoramento visual da operação do modelo em produção, foi implementado um **dashboard interativo com Streamlit**, localizado no arquivo:
> 
```
── Code/
   └── Operationalization/
       ├── streamlit_dashboard_mapa.py
       ├── streamlit_dashboard_simulacao.py
       └── streamlit_dashboard.py
```
> **Objetivos do Dashboard**
> 
> - Visualizar a distribuição dos arremessos por posição
> - Exibir métricas atualizadas de desempenho do modelo, como **Log Loss** e **F1 Score**
> - Permitir análise comparativa entre acertos e erros de arremessos
> - Oferecer **filtros interativos** por:
>   - Distância do arremesso (`shot_distance`)
>   - Período da partida (`period`)
>   - Tipo de jogo (`playoffs`)
> - Exibir visualizações como:
>   - **Mapa de arremessos**
>   - **Heatmap** da quadra
>   - Métricas agregadas e contagens
> 
> 🛠️ Funcionalidades Implementadas
> 
> - Leitura automática da base processada: `Data/Processed/predictions_prod.parquet`.
> - Cada dashboard atende a um propósito específico:
>   - `streamlit_dashboard_mapa.py`: visualização dos arremessos na quadra
>   - `streamlit_dashboard_simulacao.py`: simulação e teste de predições
>   - `streamlit_dashboard.py`: painel analítico geral com métricas agregadas
> - Filtros com Streamlit (`slider`, `checkboxes`) para refinar visualizações
> - Gráfico de dispersão com acertos e erros de arremessos sobre o mapa da quadra
> - **Heatmap de densidade** para identificar regiões de maior volume de arremessos
> - Cálculo e exibição das métricas (Log Loss e F1 Score) com base na produção
> - Layout responsivo com interface amigável e interativa
>
> ### Print Screen das telas do Dashboard e MLflow
> 
> **Streamlit - Dashboard - Mapa de Arremessos - Modelo Kobe Bryant**
>
> ![alt text](/Docs/Imagens/image_dashboard_mapa_arremessos_parte1.png)
>
> **Streamlit - Dashboard - Simulação de arremessos - Modelo Kobe Bryant**
>
> ![alt text](/Docs/Imagens/image_dashboard_simulacao_arremessos_parte1.png)
>
> ![alt text](/Docs/Imagens/image_dashboard_simulacao_arremessos_parte2.png)
> 
> ![alt text](/Docs/Imagens/image_dashboard_simulacao_arremessos_parte3.png)
> 
> **Streamlit - Dashboard Analítico - Modelo Kobe Bryant**
> 
> ![alt text](/Docs/Imagens/image_dashboard_analitico_parte1.png)
>
> ![alt text](/Docs/Imagens/image_dashboard_analitico_parte2.png)
>
> ![alt text](/Docs/Imagens/image_dashboard_analitico_parte3.png)
>
> ![alt text](/Docs/Imagens/image_dashboard_analitico_parte4.png)
>
> **MLflow - PreparacaoDados**
> 
> ![alt text](/Docs/Imagens/image_mlflow_PreparacaoDados.png)
> 
> **MLflow - Treinamento**
>
> ![alt text](/Docs/Imagens/image_mlflow_Treinamento_parte1.png)
>
> ![alt text](/Docs/Imagens/image_mlflow_Treinamento_parte2.png)
> 
> **MLflow - PipelineAplicacao**
>
> ![alt text](/Docs/Imagens/image_mlflow_PipelineAplicacao_parte1.png)
>
> ![alt text](/Docs/Imagens/image_mlflow_PipelineAplicacao_parte2.png)
>
> ![alt text](/Docs/Imagens/image_mlflow_PipelineAplicacao_parte3.png)

-------

## Rubricas e Correspondência com as Questões

**1. Desenvolver um sistema de coleta de dados usando APIs públicas**
✅ O aluno categorizou corretamente os dados?
<br>Questão: 5

✅ O aluno integrou a leitura dos dados corretamente à sua solução?
<br>Questões: 5, 6, 7

✅ O aluno aplicou o modelo em produção (servindo como API ou como solução embarcada)?
<br>Questão: 7

✅ O aluno indicou se o modelo é aderente à nova base de dados?
<br>Questão: 7-a

**2. Criar uma solução de streaming de dados usando pipelines**

✅ O aluno criou um repositório git com a estrutura de projeto baseado no Framework TDSP da Microsoft?
<br>Questão: 1

✅ O aluno criou um diagrama que mostra todas as etapas necessárias para a criação de modelos?
<br>Questão: 2

✅ O aluno treinou um modelo de regressão usando PyCaret e MLflow?
<br>Questão: 6-a

✅ O aluno calculou o Log Loss para o modelo de regressão e registrou no MLflow?
<br>Questão: 6-b

✅ O aluno treinou um modelo de árvore de decisão usando PyCaret e MLflow?
<br>Questão: 6-c

✅ O aluno calculou o Log Loss e F1 Score para o modelo de árvore de decisão e registrou no MLflow?
<br>Questão: 6-d

**3. Preparar um modelo previamente treinado para uma solução de streaming de dados**

✅ O aluno indicou o objetivo e descreveu detalhadamente cada artefato criado no projeto?
<br>Questão: 4

✅ O aluno cobriu todos os artefatos do diagrama proposto?
<br>Questão: 4

✅ O aluno usou o MLFlow para registrar a rodada "Preparação de Dados" com as métricas e argumentos relevantes?
<br>Questão: 5

✅ O aluno removeu os dados faltantes da base?
<br>Questão: 5-b

✅ O aluno selecionou as colunas indicadas para criar o modelo?
<br>Questão: 5-b

✅ O aluno indicou quais as dimensões para a base preprocessada?
<br>Questão: 5-b

✅ O aluno criou arquivos para cada fase do processamento e os armazenou nas pastas indicadas?
<br>Questão: 5 + estrutura de projeto

✅ O aluno separou em duas bases, uma para treino e outra para teste?
<br>Questão: 5-vii

✅ O aluno criou um pipeline chamado "Treinamento" no MLflow?
<br>Questão: 6

**4. Estabelecer um método de como atualizar o modelo empregado em produção**

✅ O aluno identificou a diferença entre a base de desenvolvimento e produção?
<br>Questão: 7-a

✅ O aluno descreveu como monitorar a saúde do modelo no cenário com e sem a disponibilidade da variável alvo?
<br>Questão: 7-b

✅ O aluno implementou um dashboard de monitoramento da operação usando Streamlit?
<br>Questão: 8

✅ O aluno descreveu as estratégias reativa e preditiva de retreinamento para o modelo em operação?
<br>Questão: 7-c
