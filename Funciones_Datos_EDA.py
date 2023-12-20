import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Creación de funciones para el análisis exploratorio de datos

# 0. Función para tirar la columna 'Unnamed: 0'
def drop_data(viviendas):
    viviendas.drop(columns =['Unnamed: 0', 'Fechas de publicación'], axis=1, inplace=True)
    
    return viviendas

# 1. Función para procesar datos
def limpieza_columnas(viviendas):
    viviendas['Equipamientos'] = viviendas['Equipamientos'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    viviendas['Equipamientos'] = viviendas['Equipamientos'].str.replace('[', '').str.replace(']', '')
    viviendas['Características'] = viviendas['Características'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    viviendas['Características'] = viviendas['Características'].str.replace('[', '').str.replace(']', '')
    viviendas['Consumo energético'] = viviendas['Consumo energético'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    viviendas['Consumo energético'] = viviendas['Consumo energético'].str.replace('[', '').str.replace(']', '')
    
    return viviendas

# 2. Función para separar los datos en dummies
def create_dummies(viviendas, column):
    dummies = viviendas[column].str.get_dummies(', ').astype(bool)
    
    return dummies                                          

# 3. Función para dividir la columna 'Características' en varias columnas
def div_caracteristicas(viviendas):
    viviendas['Año de construcción'] = viviendas['Características'].str.extract(r'Año de construcción: (\d{4})')
    
    viviendas['Planta'] = viviendas['Características'].str.extract(r'Planta (\d{1,2})ª')
    
    viviendas['Estado'] = viviendas['Características'].str.extract(r'Estado: (\w*)')
    viviendas['Estado'] = viviendas['Estado'].str.replace('buen', 'Buen estado')
    viviendas['Estado'] = viviendas['Estado'].str.replace('por', 'Por reformar')
    viviendas['Estado'] = viviendas['Estado'].str.replace('nuevo', 'Nuevo')
    
    viviendas['Interior/Exterior'] = viviendas['Características'].str.extract(r'Interior / exterior: (.*?)(?=,|$)')
    viviendas['Interior/Exterior'] = viviendas['Interior/Exterior'].str.strip("'")
    viviendas['Interior/Exterior'] = viviendas['Interior/Exterior'].str.replace('exterior', 'Exterior')
    viviendas['Interior/Exterior'] = viviendas['Interior/Exterior'].str.replace('interior', 'Interior')
    
    viviendas['Aire acondicionado'] = viviendas['Características'].str.extract(r'Aire acondicionado: (.*?)(?=,|$)')
    viviendas['Aire acondicionado'] = viviendas['Aire acondicionado'].str.strip("'")
    viviendas['Aire acondicionado'] = viviendas['Aire acondicionado'].str.replace('otros', 'Otros')
    
    viviendas['Tipo de calefacción'] = viviendas['Características'].str.extract(r'Combustible calefacción: (.*?)(?=,|$)')
    viviendas['Tipo de calefacción'] = viviendas['Tipo de calefacción'].str.strip("'")
    viviendas['Tipo de calefacción'] = viviendas['Tipo de calefacción'].str.replace('gas natural', 'Gas Natural')
    viviendas['Tipo de calefacción'] = viviendas['Tipo de calefacción'].str.replace('electricidad', 'Electricidad')
    viviendas['Tipo de calefacción'] = viviendas['Tipo de calefacción'].str.replace('gasóleo', 'Gasóleo')
    viviendas['Tipo de calefacción'] = viviendas['Tipo de calefacción'].str.replace('gas propano', 'Gas Propano')
    viviendas['Tipo de calefacción'] = viviendas['Tipo de calefacción'].str.replace('gas butano', 'Gas Butano')
    
    viviendas['Orientación'] = viviendas['Características'].str.extract(r'Orientado a: (.*?)$')
    viviendas['Orientación'] = viviendas['Orientación'].str.strip("'")
    viviendas['Orientación'] = viviendas['Orientación'].apply(lambda x: 1 if x == 'Sur' else 0)

    viviendas['Sistema de calefacción'] = viviendas['Características'].str.extract(r'Sistema calefacción: (.*?)(?=,|$)')
    viviendas['Sistema de calefacción'] = viviendas['Sistema de calefacción'].str.strip("'")
    viviendas['Sistema de calefacción'] = viviendas['Sistema de calefacción'].str.replace('central', 'Central')
    viviendas['Sistema de calefacción'] = viviendas['Sistema de calefacción'].str.replace('independiente', 'Independiente')
    
    return viviendas

# 4. Función para dividir la columna 'Consumo energético' en varias columnas
def div_consumo(viviendas):
    viviendas['Calificación energética'] = viviendas['Consumo energético'].str.extract(r"'(\w)'")
    viviendas['Calificación energética'] = viviendas['Calificación energética'].str.upper()

    viviendas['Consumo kWh/m² año'] = viviendas['Consumo energético'].str.extract(r"'(\d+)\s+kWh/m² año'")

    viviendas['Emisiones CO₂/m² año'] = viviendas['Consumo energético'].str.extract(r"'Emisiones', '(\d+)")
    
    return viviendas

# 5. Función para eliminar elementos de varias columnas
def remove_elements(viviendas):
    viviendas['Precios'] = viviendas['Precios'].str.replace('€', '')
    viviendas['Medidas'] = viviendas['Medidas'].str.replace('m²', '')
    viviendas['Ubicaciones'] = viviendas['Ubicaciones'].str.replace(', Madrid', '')
    
    return viviendas

# 6. Función para eliminar varias columnas
def drop_columns(df):
    df.drop(columns=['Títulos', 
                     'Equipamientos', 
                     'Características', 
                     'Consumo energético', 
                     'Descripciones', 
                     'Aire acondicionado', 
                     'Orientación', 
                     'Consumo kWh/m² año', 
                     'Emisiones CO₂/m² año'], inplace=True)
    
    return df 

# 7. Función para eliminar filas duplicadas
def eliminar_filas_duplicadas(viviendas):
    viviendas.drop_duplicates(inplace=True)
    
    return viviendas

# 8. Función para eliminar filas con valores NaN 
def remove_filas_nan(viviendas):
    viviendas.dropna(thresh = 15, inplace=True)
    
    return viviendas

# 9. Función para cambiar nombre de columnas
def nombres_columnas(viviendas):
    viviendas = viviendas.rename(columns={'Ubicaciones':'Ubicación', 
                                          'Precios': 'Precio', 
                                          'Número de habitaciones':'Habitaciones', 
                                          'Número de baños':'Baños', 
                                          'Medidas':'Superficie',  
                                          'Orientación':'Orientación Sur', 
                                          "'Aire acondicionado'":"Aire acondicionado", 
                                          "'Amueblado'":'Amueblado', 
                                          "'Armarios empotrados'":'Armarios empotrados',
                                          "'Ascensor'":'Ascensor', "'Balcón'":'Balcón',
                                          "'Calefacción'":'Calefacción',
                                          "'Garaje'":'Garaje',
                                          "'Jardín'":'Jardín',
                                          "'Piscina'":'Piscina',
                                          "'Terraza'":'Terraza'})
    
    return viviendas

# 10. Función para eliminar filas con valores 0 en la columna 'Precio'
def remove_filas_cero(viviendas):
    viviendas = viviendas[viviendas['Precio'] != 0]
    
    return viviendas

# 11. Función para cambiar el tipo de dato del DataFrame para el EDA
def change_dtype_EDA(viviendas):
    viviendas['Precio'] = pd.to_numeric(viviendas['Precio'].str.replace('.', ''), errors='coerce').astype('Int64')
    viviendas['Habitaciones'] = pd.to_numeric(viviendas['Habitaciones'].astype('float64'), errors='coerce').astype('Int64')
    viviendas['Baños'] = pd.to_numeric(viviendas['Baños'].astype('float64'), errors='coerce').astype('Int64')
    viviendas['Superficie'] = pd.to_numeric(viviendas['Superficie'].astype(str).str.replace('.', ''), errors='coerce').astype('Int64')
    viviendas['Año de construcción'] = pd.to_numeric(viviendas['Año de construcción'], errors='coerce').astype('float64')
    viviendas['Planta'] = pd.to_numeric(viviendas['Planta'].astype(str), errors='coerce').astype('float64')
    
    return viviendas

# 12. Función para crear las columnas 'Distrito' y 'Barrio'
def distrito_barrio(viviendas):
    viviendas['Barrio'] = viviendas['Ubicación'].str.split(',').str[0]
    viviendas['Distrito'] = viviendas['Ubicación'].str.split(',').str[-1].str.strip()
    viviendas.loc[viviendas['Distrito'] == 'Madrid', 'Distrito'] = np.nan
    viviendas.loc[viviendas['Barrio'].isin(['Madrid', 'Centro', 'Tetuán', 'Chamberí']), 'Barrio'] = np.nan

    viviendas.drop(['Ubicación'], axis=1, inplace=True)
    
    return viviendas

# 13. Función para ordenar las columnas de 'Distrito' y 'Barrio'
def orden(df):
    orden_columnas = ['Distrito', 'Barrio'] + [col for col in df.columns if col not in ['Distrito', 'Barrio']]
    
    return df[orden_columnas]

# 12. Función para agrupar todas las anteriores
def total_eda(df):
    df = drop_data(df)
    df = limpieza_columnas(df)
    sep_equipamientos = create_dummies(df, 'Equipamientos')
    df = pd.concat([df, sep_equipamientos], axis=1)
    df = div_caracteristicas(df)
    df = div_consumo(df)
    df = remove_elements(df)
    df = drop_columns(df)
    df = eliminar_filas_duplicadas(df)
    df = remove_filas_nan(df)
    df = nombres_columnas(df)
    df = remove_filas_cero(df)
    df = change_dtype_EDA(df)
    df = distrito_barrio(df)
    df = orden(df)
    df = df.reset_index(drop=True)
    
    return df