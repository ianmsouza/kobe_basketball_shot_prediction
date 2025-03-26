import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Patch

# Título do dashboard
st.title("Localização dos Arremessos - Kobe Bryant")

# Carrega a imagem da quadra
quadra_path = '../../Docs/Imagens/charlotte_key_zone.jpeg'
img = Image.open(quadra_path)

# Carrega os dados
df = pd.read_parquet('../../Data/Raw/dataset_kobe_dev.parquet')

# Filtro para resultado do arremesso
resultado = st.multiselect(
    'Filtrar por resultado do arremesso:',
    options=['Cesta', 'Erro', 'Desconhecido'],
    default=['Cesta', 'Erro', 'Desconhecido']
)

# Mapeamento
mapa_valores = {1: 'Cesta', 0: 'Erro', None: 'Desconhecido'}
df['resultado'] = df['shot_made_flag'].map(mapa_valores)
df['cor'] = df['shot_made_flag'].map({1: 'green', 0: 'red'})
df['cor'] = df['cor'].fillna('black')

# Filtra os dados
df_filtrado = df[df['resultado'].isin(resultado)]

# Cria o gráfico
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(-118.5, -118.05)
ax.set_ylim(33.5, 34.1)
ax.imshow(img, extent=[-118.5, -118.05, 33.5, 34.1], aspect='auto', zorder=0)

# Plota os pontos
ax.scatter(df_filtrado['lon'], df_filtrado['lat'], c=df_filtrado['cor'],
           s=10, alpha=0.7, edgecolors='k', linewidths=0.1)

ax.set_title("Localização dos Arremessos - Kobe Bryant")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Legenda personalizada
legenda = [
    Patch(color='green', label='Cesta'),
    Patch(color='red', label='Erro'),
    Patch(color='black', label='Desconhecido')
]
ax.legend(handles=legenda, loc='lower left', title='Resultado do Arremesso')

st.pyplot(fig)
