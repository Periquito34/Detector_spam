
import pandas as pd

# Leer el archivo CSV usando la codificación Windows-1252
data = pd.read_csv("spam.csv", encoding='Windows-1252')

# Imprimir los nombres de las columnas
print(data.columns)
