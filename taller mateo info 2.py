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








