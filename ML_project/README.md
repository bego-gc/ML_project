**README del Proyecto de Machine Learning (ES)**

# **Predicción de Salarios**

Este repositorio contiene código para el proyecto de ML centrado en predecir salarios basados en diversas características, utilizando la regresión XGBoost.

**Introducción**
El proyecto utiliza bibliotecas de Python como Pandas, Seaborn, Matplotlib, NumPy y scikit-learn para el procesamiento de datos, visualización y construcción del modelo.

Python 3.x
**Bibliotecas:** Pandas, Seaborn, Matplotlib, NumPy, XGBoost, scikit-learn

**Entrenamiento del Modelo**
El archivo train.py incluye código para entrenar un regresor XGBoost en el conjunto de datos preparado. Divide los datos en subconjuntos de entrenamiento y prueba, inicializa el modelo XGBoost, lo ajusta a los datos de entrenamiento y guarda el modelo entrenado utilizando Pickle con una marca de tiempo.

**Predicciones**
Utilizando el modelo entrenado, el script predict.py toma un conjunto de datos de prueba (test1.csv), realiza predicciones y guarda los resultados en un archivo CSV (predictions_test1.csv).

**Estructura de Archivos**
    src/
        data/
            Salary.csv
            test1.csv
            predictions_test1.csv
        model/production
            model_231202172502.pkl
        notebooks
            nb_borrador.ipynb
            nb_randomforest.ipynb
            nb_XGB.ipynb
        train.py
        predict.py
        memoria.ipynb

---------------------------------------------------------------------------------------------

**Machine Learning Project README (EN)**

# **Salary prediction**

This repository contains code for a Machine Learning project focusing on predicting salaries based on various features, using XGBoost regression.

**Introduction**
The project utilizes Python libraries such as Pandas, Seaborn, Matplotlib, NumPy, and scikit-learn for data processing, visualization, and model building.

Python 3.x
**Libraries:** Pandas, Seaborn, Matplotlib, NumPy, XGBoost, scikit-learn

**Model Training**
The train.py file includes code to train an XGBoost regressor on the prepared dataset. It splits the data into training and testing subsets, initializes the XGBoost model, fits it to the training data, and saves the trained model using Pickle with a timestamp.

**Predictions**
Using the trained model, the predict.py script takes in a test dataset (test1.csv), performs predictions, and saves the results in a CSV file (predictions_test1.csv).

**File Structure**
     src/
        data/
            Salary.csv
            test1.csv
            predictions_test1.csv
        model/production
            model_231202172502.pkl
        notebooks
            nb_borrador.ipynb
            nb_randomforest.ipynb
            nb_XGB.ipynb
        train.py
        predict.py
        memoria.ipynb