# Imports

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#!pip install matplotlib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr

data = pd.read_csv (r'C:\Users\bego_\Desktop\ML_project\src\data\Salary.csv')



# Feature Engineering

# Ordenar el DataFrame por la columna 'Salary' de menor a mayor
data = data.sort_values(by='Salary')
# Crear un diccionario para asignar valores numéricos a cada 'Job Title' basado en el salario
jobtitle_salary_mapping = {title: idx for idx, title in enumerate(data['Job Title'].unique(), start=1)}
# Mapear los títulos de trabajo al valor numérico basado en el salario
data['Job Title Numeric'] = data['Job Title'].map(jobtitle_salary_mapping)


# Crear un diccionario para asignar valores numéricos a cada 'Job Title' basado en el salario
jobtitle_salary_mapping = {title: idx for idx, title in enumerate(data['Job Title'].unique(), start=1)}
# Mostrar cada valor único de 'Job Title' con su valor numérico correspondiente
unique_job_titles = data['Job Title'].unique()
for title in unique_job_titles:
    job_numeric = jobtitle_salary_mapping[title]
    print(f"{title}: {job_numeric}")

# Ordenar el DataFrame por la columna 'Salary' de menor a mayor manteniendo el orden de 'Country'
data = data.sort_values(by='Salary')
# Crear un diccionario para asignar valores numéricos a cada 'Country' basado en el salario
country_salary_mapping = {country: idx for idx, country in enumerate(data['Country'].unique(), start=1)}
# Mapear los países al valor numérico basado en el salario
data['Country Numeric'] = data['Country'].map(country_salary_mapping)

# Convertir valores string de la columna 'Race' a valores numéricos
race_numeric, _ = data['Race'].factorize()
# Añadir la nueva columna 'Race Numeric' (números empezando desde 1)
data['Race Numeric'] = race_numeric + 1  # Sumar 1 para iniciar los números desde 1
# Actualizar el DataFrame original 'data' con la nueva columna 'Race Numeric'

# Mapear las string "Gender" a valores numéricos
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# DF con variables numéricas
data_numeric = data[['Age', 'Gender', 'Education Level', 'Years of Experience', 'Salary', 'Senior', 'Job Title Numeric', 'Country Numeric', 'Race Numeric']]



# Dividir en train1 y test1 para trabajar solo con el conjunto de entrenamiento train1, y mantener el conjunto de prueba test1
    # sin cambios hasta el final del proceso de aprendizaje automático

X = data_numeric[['Age', 'Gender', 'Education Level', 'Years of Experience', 'Senior', 'Job Title Numeric', 'Country Numeric', 'Race Numeric']]
y = data_numeric["Salary"] # target 

# Dividir TODOS los datos en train y test
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)


#XGB Regressor

X = data_numeric[['Age', 'Gender', 'Education Level', 'Years of Experience', 'Senior', 'Job Title Numeric', 'Country Numeric', 'Race Numeric']]
y = data_numeric["Salary"] # target 

# Dividir train1 en train2 y test2:
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train1, y_train1, test_size=0.2, random_state=42)


# Inicializar y entrenar el modelo XGBoost (XGBRegressor)
xgb = XGBRegressor()
xgb.fit(X_train2, y_train2)


# Realizar predicciones en el conjunto de prueba
predictions_xgb = xgb.predict(X_test2)



import pickle
from datetime import datetime

# Obtener la fecha actual para el nombre del modelo
fecha = datetime.now().strftime("%y%m%d%H%M%S")

# Guardar el modelo entrenado con la fecha en el nombre del archivo usando pickle
modeloXGB = f"model_{fecha}.pkl"
with open(modeloXGB, 'wb') as file:
    pickle.dump(xgb, file)

print(f"Modelo entrenado con train guardado como {modeloXGB}")
