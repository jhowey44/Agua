#!/usr/bin/env python
# coding: utf-8

# Importar librerías 

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[3]:


Modeloentreno = pd.read_csv("entrenomodelo.csv")


# In[18]:


#Agrupado = Modeloentreno.groupby(["Country Name", "Año"])


# In[22]:


#del Agrupado 


# In[16]:


# AntyBar = Modeloentreno[Modeloentreno["Country Name"] == "Australia"]


# In[26]:


#AntyBar


# ## Creación de diferencia extarcción de agua

# In[28]:


Modeloentreno['DEAD_3'] = Modeloentreno.groupby('Country Name')['EAD_3'].diff()


# In[30]:


Modeloentreno['DEAD_3'] = Modeloentreno.groupby('Country Name')['DEAD_3'].transform(lambda x: x.fillna(method='bfill'))


# ## ENTRENAR MODELO 2

# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[34]:


MEE = Modeloentreno.copy()


# In[36]:


MEE = MEE.drop(columns="Country Name")
MEE = MEE.drop(columns="Año")


# In[38]:


target = "PU_1Y"


# In[40]:


X = MEE.drop(target, axis=1)
y = MEE[target]


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)


# In[44]:


mr = RandomForestRegressor(n_estimators=100, max_depth=10, max_features=12)


# In[46]:


mr.fit(X_train, y_train)


# In[48]:


#Cor = Modeloentreno[Modeloentreno["Country Name"] == "Corea, República de"]


# In[53]:


# Obtén la importancia de las características
#feature_importances = mr.feature_importances_

# Obtén los nombres de las características
#feature_names = X_train.columns

# Combina las importancias de las características con sus nombres
#feature_importance_data = list(zip(feature_names, feature_importances))

# Ordena las características por su importancia
#feature_importance_data.sort(key=lambda x: x[1], reverse=True)

# Muestra las características más influyentes
#top_features = 23  # Puedes ajustar este valor según la cantidad de características que deseas mostrar
#for feature, importance in feature_importance_data[:top_features]:
    #print(f'Feature: {feature}, Importance: {importance}')


# ## CREAR WEB

# In[55]:


import joblib


# In[57]:


joblib.dump(mr, 'rfr_recursosnat.pkl')


# In[59]:


import streamlit as st


# 

# In[98]:


#resultados = {}
#for columna in MEE.columns:
#   min_value = MEE[columna].min()
#    max_value = MEE[columna].max()
#    resultados[columna] = {'min': min_value, 'max': max_value}
#
#print(resultados)
    


# In[100]:


# Cargar el modelo entrenado
model = joblib.load('rfr_recursosnat.pkl')

# Crear widgets para cada variable
var1 = st.slider('Extracción agua', min_value=0, max_value=1500, value=50)
var2 = st.slider('Emisiones de C02', min_value=0, max_value=8000000, value=3000000)
var3 = st.slider('Superfície KM2', min_value=120, max_value=40000000, value=5000000)
var4 = st.slider('Tierras agrícolas KM2', min_value=0, max_value=8000000, value=100000)
var5 = st.slider('Densidad poblacional', min_value=1, max_value=20000, value=5000)
var6 = st.slider('% tierras agrícolas', min_value=0, max_value=98, value=50)
var7 = st.slider('PIB', min_value=100000000, max_value=80000000000000, value=500000000000)
var8 = st.slider('Tasa fertilidad', min_value=0, max_value=8, value=2)
var9 = st.slider('Tasa muertes por 1000', min_value=1, max_value=500, value=10)
var10 = st.slider('% CER', min_value=0, max_value=100, value=50)
var11 = st.slider('Esperanza Vida Nacer', min_value=40, max_value=110, value=80)
var12 = st.slider('% Acceso Electricidad Rural', min_value=0, max_value=100, value=60)
var13 = st.slider('Tasa Mortalidad infantil', min_value=0, max_value=120, value=50)
var14 = st.slider('PIB per capita', min_value=50, max_value=500000, value=50000)
var15 = st.slider('Poblacion mayor 65', min_value=1, max_value=50, value=20)
var16 = st.slider('CUPIP', min_value=-200000, max_value=90000000000, value=1000)
var17 = st.slider('Diferencia Extraccion Agua Anual', min_value=-60, max_value=100, value=10)
var18 = st.slider('% Rentas Totales de los Recursos Naturales', min_value=0, max_value=60, value=25)
var19 = st.slider('TIE', min_value=10, max_value=200, value=50)
var20 = st.slider('Inflacion Precios Consumidor', min_value=-20, max_value=2000, value=20)
var21 = st.slider('% Acceso electricidad Urbano', min_value=0, max_value=100, value=75)
var22 = st.slider('Poblacion Mundial Total', min_value=1000000000, max_value=10000000000, value=9000000000)

# Repite para todas las 20 variables...

# Crear un DataFrame con las entradas del usuario
input_data = pd.DataFrame({
    'EAD_3': [var1],
    'pTC_3': [var2],
    'TAKM2_3': [var3],
    'pAESR_3': [var4],
    'pAESU_3': [var5],
    'pPM65_1': [var6],
    'PIBC_1': [var7],
    'IPC': [var8],
    'TMxM_1': [var9],
    'TIE_1': [var10],
    'TF_1': [var11],
    'CUPIP_1': [var12],
    'TMI_1': [var13],
    'EC2_1': [var14],
    'pRTRN_1': [var15],
    'pCER_1': [var16],
    'EVN_1': [var17],
    'PIB_1': [var18],
    'SKM2': [var19],
    'DP_2': [var20],
    'PMT_2': [var21],
    'DEAD_3': [var22],


    # Agrega las 18 variables restantes...
})

# Predecir usando el modelo
prediction = model.predict(input_data)

# Mostrar el resultado
st.write(f"Predicción: {prediction[0]}")

# Muestra el resultado en un mapa (usando folium o st.map)
import folium
lat, lon = 0,0 

m = folium.Map(location=[lat, lon], zoom_start=2)
folium.Marker(location=[lat, lon], popup=f"Predicción: {prediction[0]}").add_to(m)

st.write(m._repr_html_(), unsafe_allow_html=True)


# In[102]:


streamlit run C:\Users\jslat\anaconda3\Lib\site-packages\ipykernel_launcher.py 


# In[202]:


model.feature_names_in_

