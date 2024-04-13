import numpy as np
import pandas as pd
import scipy.io as sio


# 1. Crear una matriz de Numpy aleatoria de 4dimensiones y un sizede 1200000  

dim1=10
dim2=10
dim3=40
dim4=300

size= dim1*dim2*dim3*dim4


matriz_4d= np.random.rand(dim1,dim2,dim3,dim4)

print(matriz_4d.shape)
print(matriz_4d.size)

# 2. Crea una copia de la matriz creada en el ítem anterior(usar método copy)de solo 3 dimensiones (“Cortando una de las dimensiones”) 

dim1_nueva=dim1 
dim2_nueva=dim2 
dim3_nueva=dim3*300 
matriz_3d=matriz_4d. reshape(dim1_nueva,dim2_nueva,dim3_nueva) 
print(matriz_3d.shape) 
print(matriz_3d.size) 

#3 De la matriz 3D, muestra todos los atributos propios de dicha matriz , dimensión, tamaño, etc

forma=matriz_3d.shape #me da la forma de la matriz, es decir, el número de elem en cada dimensión
print(f"La forma de la matriz es: {forma}")
tamaño=matriz_3d.size #me da el numero total de atributos en la matriz
print(f"El tamaño de la matriz es: {tamaño}")
dimensiones=matriz_3d.ndim #Numero de dimensiones de la matriz
print(f"Las dimensiones de la matriz son: {dimensiones}")
tipoDatos=matriz_3d.dtype
print(f"Los datos de la matriz son de tipo: {tipoDatos}")
byte=matriz_3d.itemsize
print(f"Tamaño en bytes de cada elemento del array: {byte}")
byteT=matriz_3d.nbytes
print(f"Tamaño total en bytes: {byteT}")


# 4 Modificar su forma y pasarla a 2D

f=np.reshape(matriz_3d,(8000,150))
forma=f.shape #me da la forma de la matriz, es decir, el número de elem en cada dimensión
print(f"La forma de la matriz es: {forma}")
tamaño=f.size #me da el numero total de atributos en la matriz
print(f"El tamaño de la matriz es: {tamaño}")
dimensiones=f.ndim #Numero de dimensiones de la matriz
print(f"Las dimensiones de la matriz son: {dimensiones}")
tipoDatos=f.dtype
print(f"Los datos de la matriz son de tipo: {tipoDatos}")
byte=f.itemsize
print(f"Tamaño en bytes de cada elemento del array: {byte}")
byteT=f.nbytes
print(f"Tamaño total en bytes: {byteT}")

# 5.Crea una función que reciba la matriz anterior y la pase a un objeto tipo dataframe de Pandas

df= pd.DataFrame(f,columns=None)
print(df)


 # 6.Crear una función que permita cargar un archivo.mat y .csv

# 7 Crear funciones de suma, resta, multiplicación, división,logaritmo ,promedio, desviación estándar NOTA:Estas funciones deben permitir hacer estos procesos a lo largo de un eje (usando Numpy)


def sum_axis(arr, axis=None):
    return np.sum(arr, axis=axis)


def subtract_axis(arr1, arr2, axis=None):
    return np.subtract(arr1, arr2)


def multiply_axis(arr, axis=None):
    return np.prod(arr, axis=axis)


def divide_axis(arr1, arr2, axis=None):
    return np.divide(arr1, arr2)


def log_axis(arr, base=10, axis=None):
    return np.log(arr) / np.log(base)


def mean_axis(arr, axis=None):
    return np.mean(arr, axis=axis)


def std_axis(arr, axis=None):
    return np.std(arr, axis=axis)


# ejemplo
arr = np.array([[3, 4, 5], [6, 7, 8]])

# Suma a lo largo del eje 0 (columnas)
result_sum = sum_axis(arr, axis=0)
print("\nSuma a lo largo del eje 0:", result_sum)

# Resta a lo largo del eje 1 (filas)
arr2 = np.array([1, 2, 3])
result_subtract = subtract_axis(arr, arr2, axis=1)
print("\nResta a lo largo del eje 1:", result_subtract)

# Multiplicación a lo largo del eje 0 (columnas)
result_multiply = multiply_axis(arr, axis=0)
print("\nMultiplicación a lo largo del eje 0:", result_multiply)

# División a lo largo del eje 1 (filas)
arr3 = np.array([2, 2, 2])
result_divide = divide_axis(arr, arr3, axis=1)
print("\División a lo largo del eje 1:", result_divide)

# Calcular el logaritmo base 10 a lo largo de todo el arreglo
result_log = log_axis(arr, base=10)
print("Logaritmo base 10 a lo largo de todo el arreglo:\n", result_log)

# Calcular el promedio a lo largo del eje 0 (columnas)
result_mean = mean_axis(arr, axis=0)
print("Promedio a lo largo del eje 0 (columnas):", result_mean)

# Calcular la desviación estándar a lo largo del eje 1 (filas)
result_std = std_axis(arr, axis=1)
print("Desviación estándar a lo largo del eje 1 (filas):", result_std)



#8 Buscar en Kaggle un archivo .csv relacionadas con alguna patología, descargar y hacer funciones como las propuestas en el ítem anterior, pero implementándolasusando Pandas y que permitan tambienelegir columnas.

data = {'A': [4, 7, 10], 'B': [5, 8, 11], 'C': [6, 9, 12]}
df = pd.DataFrame(data)




def sum_column(df, column):
    return df[column].sum()

# Función de resta entre dos columnas
def subtract_columns(df, column1, column2):
    return df[column1] - df[column2]

# Función de multiplicación a lo largo de una columna
def multiply_column(df, column):
    return df[column].prod()

# Función de división entre dos columnas
def divide_columns(df, column1, column2):
    return df[column1] / df[column2]

# Función para calcular el logaritmo de una columna
def log_column(df, column, base=10):
    return np.log(df[column]) / np.log(base)

# Función para calcular el promedio de una columna
def mean_column(df, column):
    return df[column].mean()

# Función para calcular la desviación estándar de una columna
def std_column(df, column):
    return df[column].std()

# Cargar el conjunto de datos CSV
def cargar_csv(nombre_archivo):
    return pd.read_csv(nombre_archivo)

# ejemplo 

selected_columns = ['A', 'C']


sum_result = sum_column(df, selected_columns)
print("\nSuma de columnas seleccionadas:", sum_result)

subtract_result = subtract_columns(df, 'A', 'C')
print("\nResta entre columnas A y C:", subtract_result)

multiply_result = multiply_column(df, selected_columns)
print("\nMultiplicación de columnas seleccionadas:", multiply_result)


divide_result = divide_columns(df, 'A', 'C')
print("\nDivisión entre columnas A y C:", divide_result)


data = {'A': [4, 7, 10], 'B': [5, 8, 11], 'C': [6, 9, 12]}
df = pd.DataFrame(data)

log_result = log_column(df, selected_columns, base=10)
print("\nLogaritmo base 10 de columnas seleccionadas:\n", log_result)


mean_result = mean_column(df, selected_columns)
print("\nPromedio de columnas seleccionadas:", mean_result)

std_result = std_column(df, selected_columns)
print("\nDesviación estándar de columnas seleccionadas:", std_result)














