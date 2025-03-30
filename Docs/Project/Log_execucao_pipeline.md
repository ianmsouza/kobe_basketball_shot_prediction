(infnet-25E1_3_v2) PS C:\Users\Ian\PythonProjects\infnet-25E1_3> cd .\Code\Operationalization\
(infnet-25E1_3_v2) PS C:\Users\Ian\PythonProjects\infnet-25E1_3\Code\Operationalization> python .\main_pipeline.py    

=== Etapa 1: Preparação dos Dados ===
2025-03-30 19:15:40,842 - INFO - 🔍 Lendo os dados de desenvolvimento e produção...
2025-03-30 19:15:40,877 - INFO - 🧹 Filtrando colunas e removendo valores nulos...
2025-03-30 19:15:40,887 - INFO - ✅ Dimensão do dataset filtrado (dev): (20285, 7)
2025-03-30 19:15:40,905 - INFO - 💾 Salvando bases de treino e teste...
2025-03-30 19:15:40,910 - INFO - 📊 Registrando parâmetros e métricas no MLflow...
2025-03-30 19:15:41,164 - INFO - ✅ Pipeline de preparação de dados finalizado com sucesso.

=== Etapa 2: Treinamento do Modelo ===
2025-03-30 19:15:44,555 - INFO - 📥 Carregando bases de treino e teste...
2025-03-30 19:15:44,606 - INFO - ⚙️ Configurando o ambiente do PyCaret...
2025-03-30 19:15:45,836 - INFO - 🚀 Treinando modelo: LR
      Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold
0       0.5880  0.6183  0.5166  0.5761  0.5447  0.1706  0.1714
1       0.5907  0.6024  0.5092  0.5811  0.5428  0.1752  0.1764
2       0.5854  0.6107  0.4908  0.5770  0.5304  0.1636  0.1653
3       0.5836  0.6029  0.5037  0.5723  0.5358  0.1611  0.1622
4       0.6012  0.6230  0.5111  0.5957  0.5501  0.1958  0.1976
5       0.5607  0.5815  0.4871  0.5443  0.5141  0.1156  0.1162
6       0.5511  0.5791  0.4446  0.5356  0.4859  0.0935  0.0948
7       0.5801  0.6144  0.4659  0.5750  0.5148  0.1518  0.1544
8       0.5801  0.5987  0.4936  0.5702  0.5291  0.1538  0.1551
9       0.5498  0.5684  0.4539  0.5336  0.4905  0.0919  0.0929
Mean    0.5771  0.5999  0.4876  0.5661  0.5238  0.1473  0.1486
Std     0.0165  0.0173  0.0237  0.0198  0.0210  0.0334  0.0336
      Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold
0       0.5889  0.6181  0.5166  0.5773  0.5453  0.1723  0.1732
1       0.5915  0.6025  0.5092  0.5823  0.5433  0.1769  0.1782
2       0.5854  0.6104  0.4908  0.5770  0.5304  0.1636  0.1653
3       0.5845  0.6026  0.5037  0.5735  0.5363  0.1628  0.1639
4       0.6030  0.6230  0.5111  0.5983  0.5512  0.1992  0.2012
5       0.5616  0.5812  0.4871  0.5455  0.5146  0.1173  0.1179
6       0.5519  0.5794  0.4446  0.5367  0.4864  0.0952  0.0965
7       0.5801  0.6143  0.4641  0.5753  0.5138  0.1517  0.1544
8       0.5801  0.5985  0.4917  0.5705  0.5282  0.1537  0.1550
9       0.5498  0.5684  0.4539  0.5336  0.4905  0.0919  0.0929
Mean    0.5777  0.5998  0.4873  0.5670  0.5240  0.1484  0.1498
Std     0.0167  0.0172  0.0238  0.0201  0.0212  0.0338  0.0340
2025-03-30 19:15:52,943 - INFO - 📊 LR | Log Loss: 0.6785 | F1 Score: 0.5129
2025-03-30 19:15:52,943 - INFO - 🚀 Treinando modelo: DT
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
      Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC
Fold
0       0.5185  0.5333  0.1568  0.4857  0.2371  0.0055  0.0073
1       0.5211  0.5552  0.0775  0.4884  0.1338  0.0035  0.0065
2       0.5290  0.5535  0.1402  0.5241  0.2213  0.0249  0.0360
3       0.5273  0.5755  0.1273  0.5188  0.2044  0.0202  0.0304
4       0.5167  0.5058  0.0812  0.4632  0.1381 -0.0049 -0.0084
5       0.5335  0.5583  0.0683  0.5968  0.1225  0.0272  0.0576
6       0.5202  0.5495  0.0812  0.4835  0.1390  0.0021  0.0038
7       0.5264  0.5684  0.0497  0.5510  0.0912  0.0131  0.0310
8       0.5220  0.5607  0.0792  0.5000  0.1367  0.0069  0.0126
9       0.5163  0.5403  0.0646  0.4545  0.1131 -0.0065 -0.0124
Mean    0.5231  0.5500  0.0926  0.5066  0.1537  0.0092  0.0164
Std     0.0054  0.0188  0.0339  0.0406  0.0467  0.0112  0.0207
2025-03-30 19:15:55,259 - INFO - 📊 DT | Log Loss: 0.6903 | F1 Score: 0.1072
2025-03-30 19:15:55,259 - INFO - ✅ Modelo selecionado: LR
Transformation Pipeline and Model Successfully Saved
2025-03-30 19:15:55,381 - INFO - 💾 Modelo salvo em: ../../Data/Modeling\modelo_final.pkl
2025-03-30 19:15:55,393 - INFO - 🏁 Pipeline de treinamento finalizado.

=== Etapa 3: Aplicação em Produção ===
2025-03-30 19:15:59,191 - INFO - 📦 Carregando modelo treinado...
Transformation Pipeline and Model Successfully Loaded
2025-03-30 19:15:59,321 - INFO - 📥 Carregando dados de produção...
2025-03-30 19:15:59,349 - INFO - 🔮 Realizando predições com threshold ajustado...
Shape das probabilidades: (6426, 2)
Primeiras linhas das probabilidades:
[[0.66318238 0.33681762]
 [0.64593257 0.35406743]
 [0.68344836 0.31655164]
 [0.67778348 0.32221652]
 [0.67407707 0.32592293]]
Classes previstas pelo modelo: [0. 1.]
2025-03-30 19:15:59,385 - INFO - ✅ Resultados salvos em ../../Data/Processed\predictions_prod.parquet
2025-03-30 19:15:59,398 - INFO - 📊 Métricas calculadas: {'log_loss_prod': 0.6288800858594866, 'f1_prod': 0.1645133505598622}
               lat          lon  minutes_remaining       period     playoffs  shot_distance
count  6426.000000  6426.000000        6426.000000  6426.000000  6426.000000    6426.000000
mean     33.849664  -118.263813           4.107688     2.704171     0.137255      25.581855
std       0.082910     0.157617           3.449276     1.149971     0.344143       4.067015
min      33.253300  -118.519800           0.000000     1.000000     0.000000       0.000000
25%      33.805300  -118.421800           1.000000     2.000000     0.000000      24.000000
50%      33.835300  -118.257800           4.000000     3.000000     0.000000      25.000000
75%      33.877300  -118.115800           7.000000     4.000000     0.000000      26.000000
max      34.079300  -118.021800          11.000000     7.000000     1.000000      79.000000
Distribuição do target: 0.0    3630
1.0    1782
Name: shot_made_flag, dtype: int64

=== Etapa 4: Dashboard ===

Inicie o dashboard com: streamlit run streamlit_dashboard.py

                        streamlit run streamlit_dashboard_mapa.py

                        streamlit run streamlit_dashboard_simulacao.py

✅ Pipeline executado com sucesso!
(infnet-25E1_3_v2) PS C:\Users\Ian\PythonProjects\infnet-25E1_3\Code\Operationalization> 