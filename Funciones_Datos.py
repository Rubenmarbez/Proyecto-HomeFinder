import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Primer grupo de funciones para agrupar en una sola función ###

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

# 2. Función para eliminar filas duplicadas
def eliminar_filas_duplicadas(viviendas):
    #print("Número de filas duplicadas antes de eliminarlas:", viviendas.duplicated().sum())
    viviendas.drop_duplicates(inplace=True)
    #print("Número de filas duplicadas después de eliminarlas:", viviendas.duplicated().sum())
    
    return viviendas

# 3. Función para agrupar las funciones anteriores (0, 1, 2)
def total_limpieza(viviendas):
    viviendas = drop_data(viviendas)
    viviendas = limpieza_columnas(viviendas)
    viviendas = eliminar_filas_duplicadas(viviendas)
    
    return viviendas

                                                ### ---------------------------------- ###
                                                
# 4. Función para separar dos variables en diferentes dataframes para usar posteriormente
def sep_variables(viviendas):
    descripciones = viviendas['Descripciones']
    titulos = viviendas['Títulos']

    descripciones = pd.DataFrame(descripciones)
    titulos = pd.DataFrame(titulos)
    
    return descripciones, titulos

# 5. Función para separar los datos en dummies
def create_dummies(viviendas, column):
    dummies = viviendas[column].str.get_dummies(', ').astype(bool)
    
    return dummies
                                                ### ---------------------------------- ###
                                                
### Segundo grupo de funciones para agrupar en una sola función ###

# 6. Función para dividir la columna 'Características' en varias columnas
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

# 7. Función para dividir la columna 'Consumo energético' en varias columnas
def div_consumo(viviendas):
    viviendas['Calificación energética'] = viviendas['Consumo energético'].str.extract(r"'(\w)'")
    viviendas['Calificación energética'] = viviendas['Calificación energética'].str.upper()

    viviendas['Consumo kWh/m² año'] = viviendas['Consumo energético'].str.extract(r"'(\d+)\s+kWh/m² año'")

    viviendas['Emisiones CO₂/m² año'] = viviendas['Consumo energético'].str.extract(r"'Emisiones', '(\d+)")
    
    return viviendas

# 8. Función para eliminar elementos de varias columnas
def remove_elements(viviendas):
    viviendas['Precios'] = viviendas['Precios'].str.replace('€', '')
    viviendas['Medidas'] = viviendas['Medidas'].str.replace('m²', '')
    viviendas['Ubicaciones'] = viviendas['Ubicaciones'].str.replace(', Madrid', '')
    
    return viviendas

# 9. Función para agrupar las funciones anteriores (6, 7, 8)
def total_division(viviendas):
    viviendas = div_caracteristicas(viviendas)
    viviendas = div_consumo(viviendas)
    viviendas = remove_elements(viviendas)
    #viviendas = eliminar_filas_duplicadas(viviendas)
    
    return viviendas

                                                ### ---------------------------------- ###
                                                
### Tercer grupo de funciones para agruparlas en una sola función ###

# 10. Función para eliminar filas con valores NaN 
def remove_filas_nan(viviendas):
    viviendas.dropna(thresh = 15, inplace=True)
    
    return viviendas

# 11. Función para aplicar la moda a las columnas categóricas
def columnas_categoricas(viviendas):
    categoricas = ['Interior/Exterior', 'Tipo de calefacción', 'Sistema de calefacción', 'Calificación energética']
    for i in categoricas:
        viviendas[i].fillna(viviendas[i].mode()[0], inplace=True)
        
    return viviendas

# 12. Función para cambiar nombre de columnas
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


# 13. Función para eliminar filas con valores 0 en la columna 'Precio'
def remove_filas_cero(viviendas):
    viviendas = viviendas[viviendas['Precio'] != 0]
    
    return viviendas

# 14. Función para cambiar el tipo de dato de varias columnas
def change_dtype(viviendas):
    viviendas['Precio'] = viviendas['Precio'].astype(str).str.replace('.', '').astype('int64')
    viviendas['Habitaciones'] = viviendas['Habitaciones'].fillna(viviendas['Habitaciones'].median()).astype('int64')
    viviendas['Baños'] = viviendas['Baños'].astype('int64')
    viviendas['Superficie'] = viviendas['Superficie'].fillna(viviendas['Superficie'].median()).astype(str).str.replace('.', '').astype('int64')
    viviendas['Año de construcción'] = viviendas['Año de construcción'].fillna(viviendas['Año de construcción'].median()).astype('int64')
    viviendas['Planta'] = viviendas['Planta'].fillna(viviendas['Planta'].median()).astype('int64')
    viviendas['Consumo kWh/m² año'] = viviendas['Consumo kWh/m² año'].fillna(viviendas['Consumo kWh/m² año'].median()).astype('int64')
    viviendas['Emisiones CO₂/m² año'] = viviendas['Emisiones CO₂/m² año'].fillna(viviendas['Emisiones CO₂/m² año'].median()).astype('int64')
    
    return viviendas
    
# 15. Función para convertir en numéricas las columnas que son categóricas
def apply_categorical_columns(viviendas):
    estado_map = {'Por reformar': 0, 'Buen estado': 1, 'Nuevo': 2}
    interior_exterior_map = {'Interior': 0, 'Exterior': 1} 
    tipo_calefaccion_map = {'Gas Natural': 0, 'Electricidad': 1, 'Gasóleo': 2, 'Gas Propano': 3, 'Gas Butano': 4}
    sistema_calefaccion_map = {'Central': 0, 'Independiente': 1}
    calificacion_energetica_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G':6}

    viviendas['Estado'] = viviendas['Estado'].apply(lambda x: estado_map.get(x, x))
    viviendas['Interior/Exterior'] = viviendas['Interior/Exterior'].apply(lambda x: interior_exterior_map.get(x, x))
    viviendas['Tipo de calefacción'] = viviendas['Tipo de calefacción'].apply(lambda x: tipo_calefaccion_map.get(x, x))
    viviendas['Sistema de calefacción'] = viviendas['Sistema de calefacción'].apply(lambda x: sistema_calefaccion_map.get(x, x))
    viviendas['Calificación energética'] = viviendas['Calificación energética'].apply(lambda x: calificacion_energetica_map.get(x, x))
    
    return viviendas

# 16. Función para cambiar las ubicaciones
def replace_ubicacion(viviendas):
    viviendas['Ubicación'] = viviendas['Ubicación'].str.split().str[-1]
    
    viviendas['Ubicación'] = viviendas['Ubicación'].replace('Lineal', 'Ciudad Lineal')
    viviendas['Ubicación'] = viviendas['Ubicación'].replace('Blas', 'San Blas-Canillejas')
    viviendas['Ubicación'] = viviendas['Ubicación'].replace('Fuencarral', 'Fuencarral-El Pardo')
    viviendas['Ubicación'] = viviendas['Ubicación'].replace('Moncloa', 'Moncloa-Aravaca')
    viviendas['Ubicación'] = viviendas['Ubicación'].replace('Lineal', 'Ciudad Lineal')
    
    return viviendas

# 17. Función para crear dummies con las ubicaciones
def ubicacion_dummies (viviendas, column):
    viviendas = pd.get_dummies(viviendas, columns=[column])
    viviendas.columns = viviendas.columns.str.replace(f'{column}_', '')
    
    return viviendas

# 18. Función para agrupar las funciones anteriores (10, 11, 12, 13, 14, 15, 16, 17)
def total_changes(viviendas):
    viviendas = remove_filas_nan(viviendas)
    viviendas = columnas_categoricas(viviendas)
    viviendas = nombres_columnas(viviendas)
    viviendas = remove_filas_cero(viviendas)
    viviendas = change_dtype(viviendas)
    viviendas = apply_categorical_columns(viviendas)
    viviendas = replace_ubicacion(viviendas)
    viviendas = ubicacion_dummies(viviendas, 'Ubicación')
    
    return viviendas

# 19. Función para eliminar las columnas que no se van a usar en el modelo
def drop_columns(viviendas):
    viviendas.drop(['Orientación Sur', 
             'Tipo de calefacción', 
             'Calificación energética', 
             'Consumo kWh/m² año', 
             'Emisiones CO₂/m² año'], axis=1, inplace=True)
    
    return viviendas

# 20. Función para agrupar todas las funciones anteriores y procesar los datos de una sola vez
def process_viviendas_data(viviendas):
    viviendas = total_limpieza(viviendas)
    descripciones, titulos = sep_variables(viviendas)
    sep_equipamientos = create_dummies(viviendas, column='Equipamientos')
    viviendas = pd.concat([viviendas, sep_equipamientos], axis=1)
    viviendas = total_division(viviendas)
    viviendas2 = viviendas.copy()
    viviendas2 = viviendas2.drop(columns=['Títulos', 
                                          'Descripciones', 
                                          'Equipamientos', 
                                          'Características',  
                                          'Consumo energético', 
                                          'Aire acondicionado'], axis=1)
    viviendas2 = total_changes(viviendas2)
    viviendas2 = eliminar_filas_duplicadas(viviendas2)
    viviendas2 = drop_columns(viviendas2)
    viviendas2.reset_index(drop=True, inplace=True)
    
    return viviendas2, descripciones, titulos