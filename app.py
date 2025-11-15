
import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

#Creamos datos sintéticos realistas

np.random.seed(42)
fechas = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
n_productos = ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Auriculares']
regiones = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']

#Generamos el DataSet

data = []
for fecha in fechas:
    for _ in range(np.random.poisson(10)):    #10 Ventas promedio por día
        data.append({
            'fecha': fecha,
            'producto': np.random.choice(n_productos),
            'region': np.random.choice(regiones),
            'cantidad': np.random.randint(1, 6),
            'precio unitario': np.random.uniform(50, 1500),
            'vendedor': f'Vendedor_{np.random.randint(1, 21)}'
        })

df = pd.DataFrame(data)

#print(df)

df['venta_total'] = df['cantidad'] * df['precio unitario']
#print(df)
#print('shape del DataSet: ', df.shape)
#print(df.head(10))
#print('\nInformación General: ')
#print(df.info())
#print('\nDescripción de las variables: ')
#print(df.describe())

#1. Ventas por Mes
#def graficar_ventas(df):
#df_monthly = df.groupby(df['fecha'].dt.to_period('M'))['venta_total'].sum().reset_index()
#df_monthly['fecha'] = df_monthly['fecha'].astype(str)
#print(df_monthly)

#Configuración de la página
st.set_page_config(page_title='DashBoard de Ventas', page_icon=':bar_chart:', layout='wide')

st.title('DashBoard de Análisis de Ventas')
st.markdown('---')

#SideBar para Filtros
st.sidebar.header('Filtros')
productos_seleccionados = st.sidebar.multiselect(
    'Selecciona Productos:',
    options=df['producto'].unique(),
    default=df['producto'].unique(),
)

regiones_seleccionadas = st.sidebar.multiselect(
    'Selecciona Regiones:',
    options=df['region'].unique(),
    default=df['region'].unique(),
)

#Filtrar los datos basado en la selección
df_filtrado = df[
    (df['producto'].isin(productos_seleccionados)) &
    (df['region'].isin(regiones_seleccionadas))
]

#Ventas por Mes (con Filtros)
df_monthly = df_filtrado.groupby(df_filtrado['fecha'].dt.to_period('M'))['venta_total'].sum().reset_index()
df_monthly['fecha'] = df_monthly['fecha'].astype(str)

#_________________

fig_monthly = px.line(df_monthly, x='fecha', y='venta_total',
                        title='Tendencia de Ventas Mensuales',
                        labels={'venta_total': 'Ventas ($)', 'fecha': 'Mes'})
fig_monthly.update_traces(line=dict(width=4, color='royalblue'))
  #fig_monthly.show()

#app.graficar_ventas(df)

#2. Top productos
#def graficar_top_productos(df):
df_productos = df_filtrado.groupby('producto')['venta_total'].sum().sort_values(ascending=True)
fig_productos = px.bar(x=df_productos.values, y=df_productos.index,
                         orientation='h', title='Ventas por Producto',
                         labels={'x': 'Ventas Totales ($)', 'y': 'Producto'})
fig_productos.update_traces(marker_color='royalblue')
  #fig_productos.show()

#app.graficar_top_productos(df)

#3. análisis Geográfico
#def graficar_analisis_geografico(df):
df_regiones = df_filtrado.groupby('region')['venta_total'].sum().reset_index()
fig_regiones = px.pie(df_regiones, values='venta_total', names='region',
                        title='Distribución de Ventas por Región',
                        labels={'venta_total': 'Ventas Totales ($)'})
fig_regiones.update_traces(textposition='inside', textinfo='percent+label')
  #fig_regiones.show()

#import importlib
#import app
#importlib.reload(app)
#app.graficar_analisis_geografico(df)

#4. correlación entre variables
#def graficar_correlacion(df):
df_corr = df_filtrado[['cantidad', 'precio unitario', 'venta_total']].corr()
fig_heatmap = px.imshow(df_corr, text_auto=True, aspect='auto',
                          title='Correlación entre Variables',
                          labels=dict(x='Variables', y='Variables', color='Correlación'))
fig_heatmap.update_layout(coloraxis_colorbar_title_text='Correlación')
  #fig_heatmap.show()

#app.graficar_correlacion(df)

#5. Distribución de Ventas
#def graficar_distribucion_ventas(df):
fig_dist = px.histogram(df_filtrado, x='venta_total', nbins=50,
                          title='Distribución de Ventas Individuales')
fig_dist.update_layout(bargap=0.2)
  #fig_dist.show()

#app.graficar_distribucion_ventas(df)

#Colocar pagina

# colocar filtros 


#Métricas Principales
col1, col2, col3, col4 = st.columns(4)
with col1:
  st.metric('Ventas Totales', f"${df_filtrado['venta_total'].sum():,.0f}")
with col2:
  st.metric('Promedio por Venta', f"${df_filtrado['venta_total'].mean():,.0f}")
with col3:
  st.metric('Número de Ventas', f"{len(df_filtrado):,}")
with col4:
  crecimiento =((df_filtrado[df_filtrado['fecha'] >= '2024-01-01']['venta_total'].sum() /
                 df_filtrado[df_filtrado['fecha'] < '2024-01-01']['venta_total'].sum()) - 1) * 100
  st.metric('Crecimiento de Ventas 2024', f"{crecimiento:.1f}%")

#Layout con dos columnas
col1, col2 = st.columns(2)
with col1:
  #st.subheader('Ventas Mensuales')
  st.plotly_chart(fig_monthly, use_container_width=True)
  st.markdown('---')
  st.markdown('✅ **Análisis**: La tendencia mensual de ventas permite identificar los periodos de mayor y menor actividad comercial. Esta información es clave para planificar campañas, promociones o inventarios estratégicamente.')
  st.plotly_chart(fig_productos, use_container_width=True)
  st.markdown('---')
  st.markdown('✅ **Análisis**: El análisis por producto muestra claramente cuáles son los artículos más rentables. Los productos con mayor volumen de ventas pueden representar oportunidades de expansión o especialización comercial.')

with col2:
  st.plotly_chart(fig_regiones, use_container_width=True)
  st.markdown('---')
  st.markdown('✅ **Análisis**: La distribución geográfica revela qué regiones concentran mayor facturación. Esto ayuda a focalizar esfuerzos logísticos, comerciales y de atención al cliente en zonas clave.')
  st.plotly_chart(fig_heatmap, use_container_width=True)
  st.markdown('---')
  st.markdown('✅ **Análisis**: La matriz de correlación indica que existe una relación fuerte entre el total de venta y la cantidad, como es lógico. Esto valida la estructura del modelo de ventas y permite detectar posibles patrones de compra.')

#Gráfico completo en la parte inferior
st.plotly_chart(fig_dist, use_container_width=True)
st.markdown('---')
st.markdown("✅ **Análisis**: La distribución de ventas individuales muestra cómo se comporta el valor de cada transacción. Si la mayoría de las ventas están concentradas en un rango bajo o medio, puede considerarse una estrategia de diversificación de precios.")
