# Proyecto HomeFinder

Tema: Venta de inmuebles de segunda mano en Madrid.

Objetivo: Lograr crear una herramienta que encuentre las mejores ofertas en función de las preferencias y expectativas de cada usuario.

En este repositorio podremos encontrar los diferentes pasos que se han seguido para llevar a cabo la creación de un modelo predictivo que ayude a los usuarios a encontrar un inmueble que se adecúe a sus necesidades y a sus capacidades.

1. Parte 1 - Web Scraping: En el que encontraremos el código base y las funciones que se han usado para lograr sacar la información de un total de 9.996 anuncios de pisos dentro de la web analizada: 'https://www.yaencontre.com/venta/pisos/madrid/e-segunda-mano.com'. Durante el proceso nos encontramos que había 2.403 anuncios duplicados dentro de la web, además tuvimos que tener en consideración que no todos los anuncios están organizados de la misma manera. 
    
 2. Preprocesamiento EDA y Preprocesamiento NLP: Durante este proceso se realizó un preprocesamiento de los datos, eliminando valores duplicados y numerizando las variables, así como creando dummies de la variable 'Equipamientos' y separando 'Características' en diferentes columnas, también dividimos 'Ubicaciones' en dos nuevas variables para poder analizar, posteriormente, los 'Distritos' y 'Barrios' de cada anuncio de la web. 
En relacion al preprocesamiento NLP, se separaron las variables de 'Títulos' y 'Descripciones' que se consiguieron al realizar el scrapeo y llevamos a cabo un análisis de frecuencia de palabras para observar si existía algún patrón entre los anuncios y la forma de escribirlos para su publicación. 

3. Parte 2 - EDA: Una vez realizado el preprocesamiento de los datos, se llevó a cabo un análisis de datos exploratorios para observar como se distribuían las variables y su posible importancia en relación al precio. Durante el mismo, se pudo observar como la ubicación de cada inmueble tenía mucho peso en relación al peso. Con los valores faltantes que se encontraron el criterio a seguir fue muy similar, observar si la distribución de esas variables cambiaba en función de la manera de imputar los Nans.
    
4. Parte 3 - Modelos de entrenamiento: Por último, con todos los datos ya analizados y procesados, se llevó a cabo el entrenamiento de 3 modelos diferentes de regresión: Regresión Lineal, XG Boost y Random Forest. 
    Para la Regresión Lineal se realizó un escalado de los datos para lograr un mayor rendimiento durante el entrenamiento.
    Para el XG Boost y el Random Forest se realizó un Grid Search para encontrar los mejores parámetros con el mismo objetivo que lo mencionado anteriormente, mejorar el rendimiento y lograr mejores métricas.

    En relación a las métricas usadas para ver cómo funcionaban los modelos, se usaron el MSE, el R2_Score y el MAPE (Mean Absolute Percentage Error)

    Durante todo este proceso, se decidió reducir el número de datos, eliminado gran parte de los outliers para poder observar si habría cambios al entrenar los modelos, dejando como límite el 75% de los precios,     que equivalía a 950.000€.

    Por último, se realizaron gráficos para observar todo lo analizado anteriormente, tanto de los resultados obtenidos de los modelos con outliers y los modelos sin outliers.

En relación a todos los archivos que se comentan anteriormente, todo el código utilizado se importó desde un archivo externo donde se encuentran todas las funciones usadas en funcíon de la parte del proceso que se estuviera llevando a cabo en ese momento del proyecto.
