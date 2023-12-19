import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re

# Cargamos el link de la página web 
url = "https://www.yaencontre.com/venta/pisos/madrid/e-segunda-mano"
response = requests.get(url)
print(bool(response))
soup = BeautifulSoup(response.text, "html.parser")

#Función para obtener los links de los pisos
def get_links():
    links = []
    for i in range(1, 239):
        response = requests.get(f'https://www.yaencontre.com/venta/pisos/madrid/e-segunda-mano/pag-{i}')
        bool(response)
        soup = BeautifulSoup(response.text, "html.parser")
        titulos = soup.find('section', attrs={'class':'ThinPropertiesList'}).find_all('a', attrs={'class':'d-ellipsis'})
        for item in titulos:
            links.append(item['href'])
    return links

#Función para obtener los datos de los pisos
def get_viviendas_info(links):
    titulos = []
    ubicaciones = []
    precios = []
    numero_de_habitaciones = []
    numero_de_baños = []
    medidas_piso = []
    equipamientos = []
    caracteristicas_piso = []
    consumo_de_energia = []
    descripciones_piso = []
    fechas_de_publicacion = []

    for link in links:
        response = requests.get(f'https://www.yaencontre.com{link}')
        bool(response)
        soup = BeautifulSoup(response.text, "html.parser")

        # Conseguimos el título del anuncio de la vivienda
        titulo = soup.find('h1', attrs={'class':'details-title'})
        if titulo is not None:
            titulos.append(titulo.text)
        else:
            titulos.append(None)

        # Conseguimos la ubicación de la vivienda
        ubicacion = soup.find('h4', attrs={'class':'address-text'})
        if ubicacion is not None:
            ubicaciones.append(ubicacion.text)
        else:
            ubicaciones.append(None)

        # Conseguimos el precio de la vivienda
        precio = soup.find('span', attrs={'class':'price'})
        if precio is not None:
            precios.append(precio.text)
        else:
            precios.append(None)

        # Consguimos el número de habitaciones
        numero_habitaciones = soup.find('div', attrs={'class':'icon-room'})
        if numero_habitaciones is not None:
            numero_de_habitaciones.append(numero_habitaciones.text)
        else:
            numero_de_habitaciones.append(None)

        # Conseguimos el número de baños
        numero_baños = soup.find('div', attrs={'class':'icon-bath'})
        if numero_baños is not None:
            numero_de_baños.append(numero_baños.text)
        else:
            numero_de_baños.append(None)

        # Conseguimos la superficie de la vivienda
        medidas = soup.find('div', attrs={'class':'icon-meter'})
        if medidas is not None:
            medidas_piso.append(medidas.text)
        else:
            medidas_piso.append(None)

        # Conseguiimos el equipamiento de la vivienda
        equipamiento = soup.find('ul', attrs={'class':'outstanding-equipment'})
        if equipamiento is not None:
            equipamiento_items = [item.text.strip() for item in equipamiento.find_all('li')]
            equipamientos.append(equipamiento_items)
        else:
            equipamientos.append(None)

        # Conseguiimos las características de la vivienda
        caracteristicas = soup.find('ul', attrs={'class':'characteristics flex'})
        if caracteristicas is not None:
            caracteristicas_items = [item.text.strip() for item in caracteristicas.find_all('li')]
            caracteristicas_piso.append(caracteristicas_items)
        else:
            caracteristicas_piso.append(None)

        # Conseguimmos el consumo energético
        energia = soup.find('div', attrs={'class':'energy-certificate'})
        if energia is not None:
            energia_items = [item.text.strip() for item in energia.find_all('p')]
            if energia_items is not None:
                energia_letter = soup.find('p', class_="energy-letter pos-rel")
                if energia_letter is not None:
                    energia_items.append(energia_letter['data-rating'])
                    consumo_de_energia.append(list(filter(lambda x: x != '', energia_items)))
                else: 
                    consumo_de_energia.append(None)
            else:
                consumo_de_energia.append(None)
        else:
            consumo_de_energia.append(None)

        # Conseguimos la descripción de la vivienda
        descripcion = soup.find('div', attrs={'class':'description'})
        if descripcion is not None:
            descripciones_piso.append(descripcion.text)
        else:
            descripciones_piso.append(None)

        # Fecha de publicación del anuncio
        fecha_publicacion = soup.find('p', attrs={'class':'small-text'})
        if fecha_publicacion is not None:
            fechas_de_publicacion.append(fecha_publicacion.text)
        else:
            fechas_de_publicacion.append(None)

    df_viviendas = pd.DataFrame({'Títulos': titulos,
                                 'Ubicaciones': ubicaciones,
                                 'Precios': precios,
                                 'Número de habitaciones': numero_de_habitaciones,
                                 'Número de baños': numero_de_baños,
                                 'Medidas': medidas_piso,
                                 'Equipamientos': equipamientos,
                                 'Características': caracteristicas_piso,
                                 'Consumo energético': consumo_de_energia,
                                 'Descripciones': descripciones_piso,
                                 'Fechas de publicación': fechas_de_publicacion})

    return df_viviendas
