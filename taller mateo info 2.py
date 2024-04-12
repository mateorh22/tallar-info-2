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









