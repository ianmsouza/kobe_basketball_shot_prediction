(infnet-25E1_3_v2) PS C:\Users\Ian\PythonProjects\infnet-25E1_3\Code\Operationalization> python .\main_pipeline.py

=== Etapa 1: Preparação dos Dados ===
2025-03-26 22:06:59,382 - INFO - 🔍 Lendo os dados de desenvolvimento e produção...
2025-03-26 22:06:59,616 - INFO - 🧹 Filtrando colunas e removendo valores nulos...
2025-03-26 22:06:59,627 - INFO - ✅ Dimensão do dataset filtrado (dev): (20285, 7)
2025-03-26 22:06:59,650 - INFO - 💾 Salvando bases de treino e teste...
2025-03-26 22:06:59,672 - INFO - 📊 Registrando parâmetros e métricas no MLflow...
2025-03-26 22:07:00,100 - INFO - ✅ Pipeline de preparação de dados finalizado com sucesso.

=== Etapa 2: Treinamento do Modelo ===
2025-03-26 22:07:05,814 - INFO - 📆 Lendo base de treino...
2025-03-26 22:07:06,026 - INFO - ⚙️ Iniciando configuração do PyCaret...
                    Description            Value
0                    Session id               42
1                        Target   shot_made_flag
2                   Target type           Binary
3           Original data shape       (16228, 7)
4        Transformed data shape       (16228, 7)
5   Transformed train set shape       (11359, 7)
6    Transformed test set shape        (4869, 7)
7              Numeric features                6
8                    Preprocess             True
9               Imputation type           simple
10           Numeric imputation             mean
11       Categorical imputation             mode
12               Fold Generator  StratifiedKFold
13                  Fold Number               10
14                     CPU Jobs               -1
15                      Use GPU            False
16               Log Experiment     MlflowLogger
17              Experiment Name      Treinamento
18                          USI             c8d0
2025-03-26 22:07:07,606 - INFO - 🤖 Treinando modelo de Regressão Logística...
      Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold
0       0.5880  0.6183  0.5166  0.5761  0.5447  0.1706  0.1714
1       0.5907  0.6024  0.5092  0.5811  0.5428  0.1752  0.1764
2       0.5854  0.6107  0.4908  0.5770  0.5304  0.1636  0.1653
3       0.5836  0.6029  0.5037  0.5723  0.5358  0.1611  0.1622
4       0.6012  0.6230  0.5111  0.5957  0.5501  0.1958  0.1976
5       0.5607  0.5815  0.4871  0.5443  0.5141  0.1156  0.1162
6       0.5511  0.5791  0.4446  0.5356  0.4859  0.0935  0.0948
7       0.5801  0.6145  0.4659  0.5750  0.5148  0.1518  0.1544
8       0.5810  0.5988  0.4954  0.5711  0.5306  0.1556  0.1569
9       0.5498  0.5685  0.4539  0.5336  0.4905  0.0919  0.0929
Mean    0.5772  0.6000  0.4878  0.5662  0.5240  0.1475  0.1488
Std     0.0165  0.0173  0.0238  0.0198  0.0211  0.0334  0.0336
2025-03-26 22:07:12,223 - INFO - 🌳 Treinando modelo de Árvore de Decisão...
      Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold
0       0.5282  0.5106  0.5756  0.5049  0.5379  0.0601  0.0607
1       0.5255  0.5076  0.5461  0.5025  0.5234  0.0527  0.0528
2       0.5335  0.5056  0.6107  0.5092  0.5554  0.0730  0.0744
3       0.5176  0.5033  0.5554  0.4951  0.5235  0.0383  0.0386
4       0.5290  0.5151  0.5738  0.5057  0.5376  0.0617  0.0622
5       0.5229  0.5113  0.5683  0.5000  0.5320  0.0495  0.0499
6       0.5687  0.5643  0.6144  0.5423  0.5761  0.1405  0.1416
7       0.5352  0.5156  0.5709  0.5124  0.5401  0.0731  0.0735
8       0.5273  0.5118  0.5727  0.5049  0.5367  0.0581  0.0586
9       0.5269  0.5162  0.5572  0.5042  0.5294  0.0561  0.0564
Mean    0.5315  0.5161  0.5745  0.5081  0.5392  0.0663  0.0669
Std     0.0133  0.0166  0.0210  0.0123  0.0151  0.0266  0.0269
2025-03-26 22:07:14,932 - INFO - 📊 F1 LR: 0.5240 | F1 DT: 0.5392
Transformation Pipeline and Model Successfully Saved
2025-03-26 22:07:15,032 - INFO - ✅ Modelo final salvo em: ../../Data/Modeling\modelo_final.pkl

=== Etapa 3: Aplicação em Produção ===
2025-03-26 22:07:20,988 - INFO - 📦 Carregando modelo treinado...
Transformation Pipeline and Model Successfully Loaded
2025-03-26 22:07:21,054 - INFO - 📥 Carregando dados de produção...
2025-03-26 22:07:21,292 - INFO - 🔮 Realizando predições...
2025-03-26 22:07:21,332 - INFO - ✅ Resultados salvos em ../../Data/Processed\predictions_prod.parquet
2025-03-26 22:07:21,342 - INFO - 📊 Métricas calculadas: {'log_loss_prod': 16.337126511915613, 'f1_prod': 0.3378305450620615}

✅ Pipeline executado com sucesso!