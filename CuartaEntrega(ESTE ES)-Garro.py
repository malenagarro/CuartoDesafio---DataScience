#!/usr/bin/env python
# coding: utf-8

# # VIOLENCIA DE GÉNERO EN ARGENTINA
# 
# ## INTRODUCCIÓN 
# ### CONTEXTO EMPRESARIAL
# A lo largo de los años, la situación vulnerable de la muejer en Argentna se está haciendo visible. Por lo que, exponer la cantidad de casos y sus efectos a toda la sociadas, contribuye a la visibilizacion de las mujeres que han sufrido y las que continuarán sufirendo a causa de la desgualdad de género. Además, permitiran avanzar pasos transcendentales en materia de políticas públicas en favor de la igualdad y contra las violencias de género. Generando acciones de corto, mediano y largo plazo sustentadas para la prevención, asistencia integral y protección de aquellas mujeres que atraviesan estas situaciones de violencia. Haciendo hincapie en aquellas. Por lo que, es muy importante analizar que edades son la que mayor cantidad de casos hay y en que provincias. La informacion obtenida corresponde a aquellas comunicaciones recibdad por la Línea 144, en donde las personas que se comunican acceden a dejar sus datos para un adecuado abordaje y seguimiento. Los registros corresponden a tres sedes: Provincia de Buenos Aires, CABA y la sede de gestión nacional. Las preguntas a responder son: - En que provincias se producen más casos? - Cuales son las edades en las que se produce más violencia?
# 
# ### CONTEXTO ANALÍTICO
# Los datos ya se han recopilado y están en uso:
# 1. El archivo ¨ViolenciaGenero2.0.xlsx" que contiene el historial de los casos de violencia de género en la Argentina desde el 2020.
# 2. El archivo "HabitantesProvincia.xlsx" que contiene la cantidad de habitantes por provincia que se determinó en el Censo 2022.
# 
# 
# ## OBJETIVOS 
# En este caso, se busca realizar un análisis estadístico y su consecuente compresión de los valores con el fin de determinar las provincias y edades más afectadas.Y, finalmente, crear un modelo para predecir y prevenir los futuros casos de violencia. 
# 
# 
# Por lo tanto, se procederá de la siguiente manera: (1) se analizará los daos actuales y se evaluará las deficiencias; (2)extraer los datos de estas fuentes y realizar una limpieza de datos, EDA e ingeniería de características y (3) crear un modelo predictivo.
# 

# ## ANÁLISIS DE DATA EXISTENTE
# Antes de sumerginos en cualquier proyecto de ciencia de datos, siempre se debe evaluar los datos actuales para comprender que piezas de informacion podrían faltar. En algunos caso, no tendrá datos y tendrá que empezar de cero. En este caso, tenemos dos fuentes de datos diferentes, por lo que debemos analizar cada una de ellas individualmente y todas como un todo para averiguar como exactamente debemos complementarlas. En cada etapa, debemos tener en cuanta nuestro objetivo predecir futuros casos. Es decir, que debemos pensar la siguiente pregunta ¨Que información será útil pare predecir los futuros casos de violencia?¨

# ##### PRIMER PASO: IMPORTAMOS LIBRERIAS Y PAQUETES NECESARIAS

# In[1]:


import pandas                  as pd
from   scipy import stats
import numpy                   as np
import matplotlib.pyplot       as plt
import seaborn                 as sns
import statsmodels.formula.api as sm
import chart_studio.plotly     as py
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
# https://community.plot.ly/t/solved-update-to-plotly-4-0-0-broke-application/26526/2
import os


# ##### SEGUNDO PASO: IMPORTAMOS LA BASE DE DATOS QUE SE ENCUENTRA EN UN ARCHIVO EXCEL

# In[2]:


bd=pd.read_excel(r'C:\Users\garro\OneDrive\Escritorio\DATA SCIENCE\TRABAJO PRACTICO\ViolenciaGenero2.0.xlsx', sheet_name='casos')
bd2=pd.read_excel(r'C:\Users\garro\OneDrive\Escritorio\DATA SCIENCE\TRABAJO PRACTICO\HabitantesProvincia.xlsx', sheet_name='cantidad')


# ##### TERCER PASO: VERFICAMOS SI SE REALIZÓ LA CARGA A PARTIR DE M0STRAR LOS PRIMEROS CINCO DATOS

# In[3]:


bd.head()


# In[4]:


bd2.head()


# ##### CUARTO PASO: INSPECCIONAMOS EL DATASET PARA COMPRENDER LOS DATOS QUE TENEMOS

# In[5]:


bd.dtypes


# Las características relevantes que usaremos en este caso son:
# 
# 1. FECHA
# 2. PROVINCIA
# 3. EDAD
# 4. VINCULO_PERSONA_AGRESORA: el vinculo que tiene la vinctima con la persona agresora ("pareja", ëx pareja". "padre o tutor", "madre o tutor", "otro famiiar", "superior jerárquico", "otro")

# ## LIMPIEZA Y TRANSFORMACIÓN DE DATOS

# ##### QUINTO PASO: CAMBIAMOS EL FORMATO DE LA COLUMNA FECHA A 'DATETIME' 

# In[6]:


bd['FECHA']=pd.to_datetime(bd.FECHA, errors='coerce')
bd.head()


# ##### SEXTO PASOñ VISUALIZACIÓN DE LOS OUTLIERS Y REMOCIÓN

# ###### 6.A: VISUALIZACIÓN A TRAVÉS DEL BOXPLOT

# In[7]:


ax=sns.boxplot(x='EDAD', data=bd)


# ###### 6.B: CÁLCULO ANALÍTICO (puede ser también con la función describe())

# In[8]:


Q1=bd['EDAD'].quantile(0.25)
print('Primer cuartil', Q1)

Q3=bd['EDAD'].quantile(0.75)
print('Tercer cuartil', Q3)

IQR=Q3-Q1
print('Rango intercuartil', IQR)

mediana=bd['EDAD'].median()
print('mediana', mediana)

valor_min=bd['EDAD'].min()
print('Valor mínimo', valor_min)

valor_max=bd['EDAD'].max()
print('Valor máximo', valor_max)

Valor_BI=(Q1-1.5*IQR)
print('Valor_BI', Valor_BI)

Valor_BS=(Q3+1.5*IQR)
print('Valor_BS', Valor_BS)


# ###### 6.C: ELIMINACIÓN DE OUTLIERS
# Con esta función, se detecta el outlier y se intercambia con el valor de la media.

# In[9]:


def clean_age(age):
    if age>=Valor_BI and age<=Valor_BS:
        return age
    else:
            return mediana
bd['age_clean'] = bd['EDAD'].apply(clean_age)

# Check out the new column and make sure it looks right

print("'EDADES'")
print("Valor mínimo: ", bd["age_clean"].min())
print("Valor máximo: ", bd["age_clean"].max())


# ###### 6.D: VISUALIZACIÓM DE DATOS SIN ULIERS

# In[10]:


ax=sns.boxplot(x='age_clean', data=bd)


# ###### 6.E: INTERCAMBIO DE VALORES NaN POR LA MEDIA

# In[11]:


bd['age_clean'].fillna(value=mediana, inplace=True)
bd.head(10)


# ## VISUALIZACIÓN DE LOS DATOS Y BÚSQUEDA DE PATRONES

# ##### SEPTIMO PASO: VISUALIZACIÓN DE LA VARIACIÓN DE CANT DE CASOS EN EL TIEMPO

# ###### 7.A: PARA FACILITAR EL ANÁLISIS AGRUPAMOS LAS FECHAS DE MANERA MENSUAL (no diaria)

# In[12]:


bd['FECHA'] = pd.to_datetime(bd['FECHA'], errors='coerce')
bd['FECHA_MES']=bd.FECHA.dt.to_period('M')
bd3=bd.groupby('FECHA_MES', as_index=False).sum()
bd3.head()


# In[13]:


bd3['FECHA_MES'] = bd3['FECHA_MES'].astype('str')


# ###### 7.B: REALIZAMOS UN GRÁFICO DE LINEAS

# In[14]:


ax = sns.lineplot(data=bd3, x="FECHA_MES", y="CASO")
ax.set(xlabel='Tiempo', ylabel='Cantidad de casos', title='Variación en el tiempo')


# ##### OCTAVO PASO: REALIZAMOS UN HISTOGRAMA EN EL QUE PODAMOS ANALIZAR LA DISTRIBUCIÓN DE LAS EDADES EN CADA PROVINCIA
# 

# In[15]:


sns.displot(data=bd, x="EDAD", hue="PROVINCIA", multiple="stack")


# ##### NOVENO PASO: GRÁFICAMOS CANTIDAD DE CASOS POR PROVINCIA:
# ###### 9.A: CALCULAMOS LA CANTIDAD DE CASOS TOTALES POR PROVINCIA

# In[16]:


serie_provincia=bd.PROVINCIA.value_counts()
serie_provincia


# ###### 9.B: GRAFICAMOS

# In[17]:


fig, ax= plt.subplots()
ax.barh(serie_provincia.index, serie_provincia, label='Casos totales')
ax.legend(loc='upper right')
ax.set_title('Cantidad de casos en cada provincia')
ax.set_ylabel('Provincias')
ax.set_xlabel('Cantidad de casos')


# ###### 9.C: PARA MEJORAR LA VISUALIZACIÓN PODEMOS REALIZAR UNA CATEGORIZACION DE LAS PROVINCIAS EN REGIONES

# In[18]:


# REALIZAMOS UNA COPIA DE LA COLUMNA PROVINCIA PARA PRESERVAR LOS DATOS ORIGNALES.
bd['REGION'] = bd['PROVINCIA']

# EN LA NUEVA COLUMNA ASIGNAMOS UNA NUEVA CATEGORIA
PAMPEANA = ['Ciudad Autónoma de Buenos Aires', 'Buenos Aires', 'Córdoba', 'Entre Ríos', 'La Pampa','Santa Fe']
NOA = ['Catamarca', 'Jujuy', 'La Rioja', 'Salta', 'Santiago del Estero', 'Santiago Del Estero', 'Tucumán']
NEA = ['Corrientes', 'Chaco', 'Formosa', 'Misiones'] 
CUYO = ['Mendoza', 'San Luis', 'San Juan']
PATAGONIA = ['Chubut', 'Neuquén', 'Río Negro', 'Santa Cruz', 'Tierra del Fuego, Antártida e Islas del Atlántico Sur']

bd['REGION'] = bd['REGION'].apply(lambda x:"PAMPEANA" if x in PAMPEANA else x)
bd['REGION'] = bd['REGION'].apply(lambda x:"NOA" if x in NOA else x)
bd['REGION'] = bd['REGION'].apply(lambda x:"NEA" if x in NEA else x)
bd['REGION'] = bd['REGION'].apply(lambda x:"CUYO" if x in CUYO else x)
bd['REGION'] = bd['REGION'].apply(lambda x:"PATAGONIA" if x in PATAGONIA else x)

# CREAMOS UNA SERIE DE LAS REGIONAS CONTANDO LOS CASOS
serie_regiones=bd.REGION.value_counts()
serie_regiones

print(serie_regiones)


# ###### 9.D: GRAFICAMOS

# In[19]:


fig, ax= plt.subplots()
ax.barh(serie_regiones.index, serie_regiones, label='Casos totales')
ax.legend(loc='upper right')
ax.set_title('Cantidad de casos en cada provincia')
ax.set_ylabel('Cantidad de casos')
ax.set_xlabel('Regiones')


# ##### DECIMO PASO:REALIZAMOS UNA ESTADICTICA DE EDADES POR PROVINCIA

# In[20]:


sns.boxplot(x=bd.REGION, y= bd.EDAD)
plt.title('Boxplot comparativo entre las regiones de Argentina en funcion de la edad')
plt.xlabel('Región')
plt.ylabel('Edad')


# Por lo tanto, gracias a estos gráficos podemos realizar las primeras conclusiones:
# 
# * la mayor cantidad de casos se da en casos con victimas de un rango de edad de entre 27-43 años, siendo la edad en la que se concentra el 50% de los casos a los 34 años
# * la provincia con mayor cantidad de casos es Buenos Aires, mientras que la de menor cantidad es La Pampa
# 
# AVISO IMPORTANTE! PARA UNA MAYOR CARACTERIZACIÓN DE LOS DATOS DEBEMOS REALIZAR UNA NORMALIZACIÓN DE ESTOS. A partir de dividir esta cantidad de casos por provincia por su respectiva cantidad de habitantes (queda pendiente para la proxima entrega)

# In[21]:


model1 = 'EDAD~REGION'
lm1   = sm.ols(formula = model1, data = bd).fit()
print(lm1.summary())


# ##### ONCEAVO PASO:ANALIZAMOS EL VINCULO DEL AGRESOR CON LA VICTIMA

# ###### 11.A: REALIZAMOS UN RECUENTO DE CASOS POR AGRESOR

# In[22]:


vinculo=bd.groupby('VINCULO_PERSONA_AGRESORA')
cant=bd.groupby(bd.VINCULO_PERSONA_AGRESORA)['CASO'].count()
cant


# ###### 11.B: REALIZAMOS UN GRÁFICO DE TORTA PARA VISUALIZACIÓN DE DATOS

# In[23]:


fig1, ax1 = plt.subplots()
#Creamos el grafico, añadiendo los valores
vinculo=['Ex pareja', 'Madre o tutor', 'Otro', 'Otro familiar', 'Padre o tutor', 'Pareja', 'Superior jerárquico']
ax1.pie(cant, labels=vinculo, autopct='%1.1f%%', shadow=True, startangle=90)
#señalamos la forma, en este caso 'equal' es para dar forma circular
ax1.axis('equal')
plt.title('Distribución de vinculo de agresor')
#plt.legend()
plt.savefig('grafica_pastel.png')
plt.show()


# ###### 11.C: REALIZAMOS HISTOGRAMA QUE ANALICE EL VINCULO DEL AGRESOR EN FUNCIÓN A LA EDAD DE LA VICTIMA
# ESTO PERMITIRA CONOCER PARA CUALES SON LAS EDADES MAS VULNERABLES PARA CADA TIPO DE ÄGRESOR

# In[24]:


plt.figure()
# Figure -level
ax = sns.displot(data=bd, x='EDAD', kind='kde', hue='VINCULO_PERSONA_AGRESORA', fill=True)
ax.set(xlabel='Edad', ylabel='Densidad', title='Distribución  de edades en función a vincuo con agresor')


# ###### 11.D: GRAFICAMOS BOXPLOT PARA ANLIZAR LAS EDADES

# In[25]:


sns.boxplot(x=bd.VINCULO_PERSONA_AGRESORA, y= bd.EDAD)
plt.title('Boxplot comparativo vinculo de persona en funcion de la edad')
plt.xlabel('Vinculo de persona agresora')
plt.ylabel('Edad')


# In[26]:


model2 = 'EDAD~VINCULO_PERSONA_AGRESORA'
lm1   = sm.ols(formula = model2, data = bd).fit()
print(lm1.summary())


# ## CONCLUSIONES
# Usamos las técnicas de regresión lineal para determinar si existía o no una relaciones entre diferentes variables
# en los casos de violencia de género. 
# En primer lugar, buscamos determinar la cantidad de casos por provincia y cual su varianza. Esto nos permite definir
# cuales son los rangos de edad más vulnerables y en un futuro, enfatizar las acciones legislativas y de asistencia a 
# esas edades. En este caso, observamos que el 50% de los casos se acumulaban entre los 27 (25%) a 43 (75%) años, siendo
# el 50% los 34 años.
# En segundo lugar, realizamos un grfico de barras que indicaba la cantidad de casos por provincia. Esto permite darnos 
# una idea de cuales son las provincias más vulerables y, por lo tanto, crear una mayor cantidad de centros de asistencia
# a la mujer. En este caso, observamos que la provincia con mayor cantidad de casos es Buenos Aires, mientras que la de 
# menor cantidad es La Pampa. 
# Luego, para facilitar la visualización de los datos realizamos una categorización de las provincias en regiones (Pampeana,
# NOA, NEA, Cuyo y Patagonia).Además, buscamos identificar la media de las edades de las victimas nn cada región. 
# En tercer lugar, determinamos la cantidad de casos en función al vincula del agresor con la víctima y, cuales son las edades
# más afectadas en cada tipo de vínculo. Observamos que las victimas cuyo agresor es un familiar (madre, padre, tutor u ptro
# familiar) son entre los 11 a 20 años. Mientras que, aquellas en las que el agresor es la pareja o ex pareja son de entre 18 
# a 45 años. 
#               

# In[ ]:




