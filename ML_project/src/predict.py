import pandas as pd
from xgboost import XGBRegressor
import pickle

# Cargar el modelo entrenado (model_231202172502.pkl)
modelo_entrenado = r'C:\Users\bego_\Desktop\ML_project\src\model\production\model_231202172502.pkl'
with open(modelo_entrenado, 'rb') as file:
    modelo = pickle.load(file)

# Cargar los datos de test1 (test1.csv)
test1 = pd.read_csv(r'C:\Users\bego_\Desktop\ML_project\src\data\test1.csv')

# Realizar predicciones en el conjunto test1
columnas_prediccion = ['Age', 'Gender', 'Education Level', 'Years of Experience', 'Senior', 'Job Title Numeric', 'Country Numeric', 'Race Numeric']
predicciones_test1 = modelo.predict(test1[columnas_prediccion])

# Crear un DataFrame con las predicciones
df_predicciones_test1 = pd.DataFrame({'Predicciones': predicciones_test1})

# Guardar las predicciones en un archivo CSV en la carpeta 'data'
ruta_predicciones_test1_csv = (r'C:\Users\bego_\Desktop\ML_project\src\data\predicciones_test1.csv')
df_predicciones_test1.to_csv(ruta_predicciones_test1_csv, index=False)
