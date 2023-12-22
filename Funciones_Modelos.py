import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

from Funciones_Datos import *

import warnings
warnings.filterwarnings('ignore')

### Funciones para modelos con Outliers ###

# Función para modelo de Regresión Lineal
def lr_model(viviendas2):
    # Separamos los datos en X (variables independientes) e y (variable dependiente)
    X = viviendas2.drop('Precio', axis=1)
    y = viviendas2['Precio']

    # Separamos los datos en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Hacemos un escalado de los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Creamos el modelo
    model = LinearRegression()

    # Fit y precicción del modelo
    model.fit(X_train_scaled, y_train)
    y_pred_lr = model.predict(X_test_scaled)

    # Calculamos el Mean Squared Error (MSE)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    mse_squared_lr = np.sqrt(mse_lr).round(2)

    # Calculamos el Mean Absolute Percentage Error (MAPE)
    mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr).round(2)
    
    # Calculamos el R-squared (R2) score
    r2_lr = r2_score(y_test, y_pred_lr)

    # Resultados
    print('Mean Squared Error:', mse_lr)
    print('Mean Squared Error Sqrt:', mse_squared_lr)
    print('Mean Absolute Percentage Error:', mape_lr)
    print('R-squared score:', r2_lr)
    
    return y_pred_lr, mse_lr, mse_squared_lr, mape_lr, r2_lr

# Función para GridSearch para XGBoost
def run_grid_search_xgb(X_train, y_train):
    # Parámetros GridSearchCV para XGBoost
    param_grid = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.8, 0.9, 1],
        'colsample_bytree': [0.8, 0.9, 1],
        'gamma': [0.01, 0.1, 1],
        'reg_alpha': [0.1, 0.5],
        'reg_lambda': [0.1, 0.5],
        'random_state': [42]
    }

    # Creamos el modelo XGBoost
    model = xgb.XGBRegressor(objective='reg:squarederror')

    # Ejectuamos el GridSearchCV
    grid_search_xgb = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search_xgb.fit(X_train, y_train)

    # Conseguimos e imprimimos los mejores parámetros y el mejor score
    best_params_xgb = grid_search_xgb.best_params_
    best_score_xgb = np.sqrt(-grid_search_xgb.best_score_)
    best_model_xgb = grid_search_xgb.best_estimator_
    
    print("Best Parameters:", best_params_xgb)
    print("Best Score (RMSE):", best_score_xgb)
    
    return best_model_xgb


# Función para GridSearch de Random Forest
def run_grid_search_rf(X_train, y_train):
    # Parámetros GridSearchCV para Random Forest
    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [5, 7, 9],
        'min_samples_split': [10],
        'min_samples_leaf': [4],
        'max_features': [0.7],
        'random_state': [42]
    }

    # Creamos el modelo Random Forest
    model = RandomForestRegressor()

    # Ejectuamos el GridSearchCV
    grid_search_rf = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search_rf.fit(X_train, y_train)

    # Conseguimos e imprimimos los mejores parámetros y el mejor score
    best_params_rf = grid_search_rf.best_params_
    best_score_rf = np.sqrt(-grid_search_rf.best_score_)
    best_model_rf = grid_search_rf.best_estimator_
    
    print("Best Parameters:", best_params_rf)
    print("Best Score (RMSE):", best_score_rf)
    
    return best_model_rf

### Graficamos los resultados ###
                                            
# Graficamos los resultados predichos vs los reales
def plot_results(y_test, y_pred_lr, y_pred_xgb, y_pred_rf):
    # Gráfico de los resultados predichos vs reales para Linear Regression
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred_lr, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel('Precios (€)')
    plt.ylabel('Valores Predichos')
    plt.title('Linear Regression')

    # Gráfico de los resultados predichos vs reales para XGBoost
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_pred_xgb, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel('Precios (€)')
    plt.ylabel('Valor Predichos')
    plt.title('XGBoost')

    # Gráfico de los resultados predichos vs reales para Random Forest
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, y_pred_rf, color='orange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel('Precios (€)')
    plt.ylabel('Valor Predichos')
    plt.title('Random Forest')

    plt.tight_layout()
    plt.show()
    
# Graficamos el error  de los resultados predichos vs los reales
def plot_error(y_test, y_pred_lr, y_pred_xgb, y_pred_rf):
    # Calculamos el error entre los valores predichos y los reales
    error_lr = y_pred_lr - y_test
    error_xgb = y_pred_xgb - y_test
    error_rf = y_pred_rf - y_test

    # Gráfico de los errores para Regresión Lineal
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, error_lr, color='blue')
    plt.xlabel('Precios (€)')
    plt.ylabel('Error')
    plt.title('Linear Regression')

    # Gráfico de los errores para XGBoost
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, error_xgb, color='green')
    plt.xlabel('Precios (€)')
    plt.ylabel('Error')
    plt.title('XGBoost')

    # Gráfico de los errores para Random Forest
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, error_rf, color='orange')
    plt.xlabel('Precios (€)')
    plt.ylabel('Error')
    plt.title('Random Forest')

    plt.tight_layout()
    plt.show()
    
# Graficamos el porcentaje de error de los resultados predichos vs los reales
def plot_porcentaje_error(y_test, y_pred_lr, y_pred_xgb, y_pred_rf):
    # Gráfico de los errores para Regresión Lineal
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, np.abs(y_pred_lr - y_test) / y_test, color='blue')
    plt.xlabel('Precios (€)')
    plt.ylabel('Porcentaje de Error Absoluto')
    plt.title('Linear Regression')
    
    # Gráfico de los errores para XGBoost
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, np.abs(y_pred_xgb - y_test) / y_test, color='green')
    plt.xlabel('Precios (€)')
    plt.ylabel('Porcentaje de Error Absoluto')
    plt.title('XGBoost')
    
    # Gráfico de los errores para Random Forest
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, np.abs(y_pred_rf - y_test) / y_test, color='orange')
    plt.xlabel('Precios (€)')
    plt.ylabel('Porcentaje de Error Absoluto')
    plt.title('Random Forest')
    
    plt.tight_layout()
    plt.show()

### Funciones para modelos sin Outliers ###
                                
# Función para modelo de Regresión Lineal sin Outliers        
def lr_model_out(viviendas_cleaned):
# Separamos los datos en X (variables independientes) e y (variable dependiente)
    X = viviendas_cleaned.drop('Precio', axis=1)
    y = viviendas_cleaned['Precio']

    # Separamos los datos en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hacemos un escalado de los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Creamos el modelo
    model = LinearRegression()

    # Fit y precicción del modelo
    model.fit(X_train_scaled, y_train)
    y_pred_lr_out = model.predict(X_test_scaled)

    # Calculamos el Mean Squared Error (MSE)
    mse_lr_out = mean_squared_error(y_test, y_pred_lr_out)
    mse_squared_lr_out = np.sqrt(mse_lr_out).round(2)

    # Calculamos el Mean Absolute Percentage Error (MAPE)
    mape_lr_out = mean_absolute_percentage_error(y_test, y_pred_lr_out).round(2)
    
    # Calculamos el R-squared (R2) score
    r2_lr_out = r2_score(y_test, y_pred_lr_out)

    # Resultados
    print('Mean Squared Error:', mse_lr_out)
    print('Mean Squared Error Sqrt:', mse_squared_lr_out)
    print('Mean Absolute Percentage Error:', mape_lr_out)
    print('R-squared score:', r2_lr_out)

    return y_pred_lr_out, mse_lr_out, mse_squared_lr_out, mape_lr_out, r2_lr_out

# Función para GridSearch para XGBoost
def run_grid_search_xgb_out(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.8, 0.9, 1],
        'colsample_bytree': [0.8, 0.9, 1],
        'gamma': [0.01, 0.1, 1],
        'reg_alpha': [0.1, 0.5],
        'reg_lambda': [0.1, 0.5],
        'random_state': [42]
    }

    model = xgb.XGBRegressor(objective='reg:squarederror')

    grid_search_xgb_out = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search_xgb_out.fit(X_train, y_train)

    best_params_xgb_out = grid_search_xgb_out.best_params_
    best_score_xgb_out = np.sqrt(-grid_search_xgb_out.best_score_)
    best_model_xgb_out = grid_search_xgb_out.best_estimator_
    
    print("Best Parameters:", best_params_xgb_out)
    print("Best Score (RMSE):", best_score_xgb_out)
    
    return best_model_xgb_out

# Función para GridSearch para Random Forest
def run_grid_search_rf_out(X_train, y_train):
    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [5, 7, 9],
        'min_samples_split': [10],
        'min_samples_leaf': [4],
        'max_features': [0.7],
        'random_state': [42]
    }

    model = RandomForestRegressor()

    grid_search_rf_out = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search_rf_out.fit(X_train, y_train)

    best_params_rf_out = grid_search_rf_out.best_params_
    best_score_rf_out = np.sqrt(-grid_search_rf_out.best_score_)
    best_model_rf_out = grid_search_rf_out.best_estimator_
    
    print("Best Parameters:", best_params_rf_out)
    print("Best Score (RMSE):", best_score_rf_out)

    return best_model_rf_out

### Graficamos los resultados sin Outliers ###

# Graficamos los resultados predichos vs los reales                                        
def plot_results_out(y_test, y_pred_lr_out, y_pred_xgb_out, y_pred_rf_out):
# Gráfico de los resultados predichos vs reales para Linear Regression
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred_lr_out, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel('Precios (€)')
    plt.ylabel('Valores Predichos')
    plt.title('Linear Regression')

    # Gráfico de los resultados predichos vs reales para XGBoost
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_pred_xgb_out, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel('Precios (€)')
    plt.ylabel('Valor Predichos')
    plt.title('XGBoost')

    # Gráfico de los resultados predichos vs reales para Random Forest
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, y_pred_rf_out, color='orange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel('Precios (€)')
    plt.ylabel('Valor Predichos')
    plt.title('Random Forest')

    plt.tight_layout()
    plt.show()
 
# Graficamos el error  de los resultados predichos vs los reales
def plot_error_out(y_test, y_pred_lr_out, y_pred_xgb_out, y_pred_rf_out):
    # Calculamos el error  entre los valores predichos y los reales
    error_lr_out = y_pred_lr_out - y_test
    error_xgb_out = y_pred_xgb_out - y_test
    error_rf_out = y_pred_rf_out - y_test

    # Gráfico de los errores para Regresión Lineal
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, error_lr_out, color='blue')
    plt.xlabel('Precios (€)')
    plt.ylabel('Error')
    plt.title('Linear Regression')

    # Gráfico de los errores para XGBoost
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, error_xgb_out, color='green')
    plt.xlabel('Precios (€)')
    plt.ylabel('Error')
    plt.title('XGBoost')

    # Gráfico de los errores para Random Forest
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, error_rf_out, color='orange')
    plt.xlabel('Precios (€)')
    plt.ylabel('Error')
    plt.title('Random Forest')

    plt.tight_layout()
    plt.show()
    
# Graficamos el porcentaje de error de los resultados predichos vs los reales
def plot_porcentaje_error_out(y_test, y_pred_lr_out, y_pred_xgb_out, y_pred_rf_out):
    # Gráfico de los errores para Regresión Lineal
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, np.abs(y_pred_lr_out - y_test) / y_test, color='blue')
    plt.xlabel('Precios (€)')
    plt.ylabel('Porcentaje de Error Absoluto')
    plt.title('Linear Regression')
    
    # Gráfico de los errores para XGBoost
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, np.abs(y_pred_xgb_out - y_test) / y_test, color='green')
    plt.xlabel('Precios (€)')
    plt.ylabel('Porcentaje de Error Absoluto')
    plt.title('XGBoost')
    
    # Gráfico de los errores para Random Forest
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, np.abs(y_pred_rf_out - y_test) / y_test, color='orange')
    plt.xlabel('Precios (€)')
    plt.ylabel('Porcentaje de Error Absoluto')
    plt.title('Random Forest')
    
    plt.tight_layout()
    plt.show()