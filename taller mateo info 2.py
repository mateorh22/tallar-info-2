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









