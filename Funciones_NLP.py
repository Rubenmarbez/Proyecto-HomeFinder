import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import re

import itertools
import nltk
from wordcloud import WordCloud
from nltk.stem import PorterStemmer

# 1. Función para eliminar NaNs y resetear el índice de un DataFrame
def clean_viviendas_nlp(viviendas_nlp):
    viviendas_nlp.dropna(inplace=True)
    viviendas_nlp.drop_duplicates(inplace=True)
    viviendas_nlp.reset_index(drop=True, inplace=True)
    
    return viviendas_nlp

# 2. Función para tokenizar y graficar la frecuencia de las palabras de las descripciones
def get_descripciones(viviendas_nlp, stopwords):
    descripciones_all = []
    for i in range(viviendas_nlp.shape[0]):
        descripciones = viviendas_nlp.Descripciones[i]
        descripciones = nltk.tokenize.RegexpTokenizer("[\w]+").tokenize(descripciones)
        descripciones = [word.lower() for word in descripciones if word.lower() not in stopwords and not word.isdigit()]
        descripciones_all.append(descripciones)
        
    descripciones_all = list(itertools.chain(*descripciones_all))
    freq_descripciones = nltk.FreqDist(descripciones_all)

    df_descripciones = pd.DataFrame(list(freq_descripciones.items()), columns=["Palabra", "Frecuencia"])
    df_descripciones.sort_values('Frecuencia', ascending=False, inplace=True)
    df_descripciones.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(15, 8))
    plot = sns.barplot(x=df_descripciones.iloc[:30].Palabra, y=df_descripciones.iloc[:30].Frecuencia)
    for item in plot.get_xticklabels():
        item.set_rotation(90)
    plt.show()

    return df_descripciones

# 3. Función para tokenizar y graficar la frecuencia de las palabras de los títulos
def get_titulos(viviendas_nlp, stopwords):
    titulos_all = []
    for i in range(viviendas_nlp.shape[0]):
        titulos = viviendas_nlp.Títulos[i] 
    # Le pido que solo saque las palabras y las tokenice
        titulos = nltk.tokenize.RegexpTokenizer("[\w]+").tokenize(titulos) 
    # Le pido que me devuelva las palabras en minúscula y que no me devuelva los números
        titulos = [word.lower() for word in titulos if word.lower() not in stopwords and not word.isdigit()]
        titulos_all.append(titulos)
        
    titulos_all = list(itertools.chain(*titulos_all))
    freq_titulos = nltk.FreqDist(titulos_all)

    df_titulos = pd.DataFrame(list(freq_titulos.items()), columns=["Palabra", "Frecuencia"])
    df_titulos.sort_values('Frecuencia', ascending=False, inplace=True)
    df_titulos.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(15, 8))
    plot = sns.barplot(x=df_titulos.iloc[:30].Palabra, y=df_titulos.iloc[:30].Frecuencia)
    for item in plot.get_xticklabels():
        item.set_rotation(90)
    plt.show()

    return df_titulos

# 4. Función WordCloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(10, 10))
    # Display image
    plt.imshow(wordcloud)
    # No axis details
    plt.axis("off")