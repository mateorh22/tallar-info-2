import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch


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

 #def load(archivo):
  #if archivo.endswith('.mat'):
   # return sio.loadmat(archivo, variable_names=['nombre_variable'])
  #elif archivo.endswith('.csv'):
   # return pd.read_csv(archivo)
  #else:
   # raise ValueError("Formato de archivo no soportado")

#archivo = "/content/E00001.mat"
#loaded_data= load(archivo)
#loaded_data
#NO me funciono profe


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

# 9 Usar matplotlib para graficar la señal del archivo mat del punto 6 y crear funciones para graficar histogramas, stems, barras, pies 
matrix = np.array(f)


plt.imshow(matrix, cmap='viridis')  # cmap es un mapa de colores opcional
plt.colorbar()  # Agregar una barra de colores para indicar los valores
plt.title("Matriz 2D")  # Agregar un título a la gráfica
plt.xlabel("Eje X")  # Etiqueta del eje X
plt.ylabel("Eje Y")  # Etiqueta del eje Y

plt.show()


# 10 Las funciones de graficación deben pedir al usuario los títulos de gráficos y los ejes, activar leyendas , activar la cuadricula.

def plot_histogram_subplot(data, bins=10, title="", xlabel="", ylabel="", orientation='vertical'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) if orientation == 'horizontal' else plt.subplots(2, 1, figsize=(5, 12))
    axes[0].hist(data, bins=bins, edgecolor='k')
    axes[0].set_title(title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    if orientation == 'horizontal':
        plt.tight_layout()
    plt.show()

# Ejemplo de uso
data = [1, 5, 6, 4, 4, 4, 2, 7, 6, 6, 6, 6]
plot_histogram_subplot(data, bins=5, title="Histograma de Ejemplo", xlabel="Valores", ylabel="Frecuencia", orientation='vertical')

#Stem
def plot_stem_subplot(data, title="", xlabel="Tallos", ylabel="Hojas", orientation='vertical'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) if orientation == 'horizontal' else plt.subplots(2, 1, figsize=(5, 12))
    stems, leaves = zip(*[(int(str(val)[:-1]), int(str(val)[-1])) for val in data])
    axes[0].stem(stems, leaves, use_line_collection=True)
    axes[0].set_title(title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    if orientation == 'horizontal':
        plt.tight_layout()
    plt.show()

data = [14, 27, 37, 52, 58, 73, 81, 89, 90]
plot_stem_subplot(data, title="Gráfico de Tallos y Hojas de Ejemplo", orientation='vertical')

def plot_bar_subplot(data, labels, title="", xlabel="", ylabel="", orientation='vertical'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) if orientation == 'horizontal' else plt.subplots(2, 1, figsize=(5, 12))
    axes[0].bar(labels, data)
    axes[0].set_title(title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    if orientation == 'horizontal':
        plt.tight_layout()
    plt.show()


data = [10, 15, 20, 25, 30]
labels = ['A', 'B', 'C', 'D', 'E']
plot_bar_subplot(data, labels, title="Gráfico de Barras de Ejemplo", xlabel="Categorías", ylabel="Valores", orientation='vertical')


def plot_pie_subplot(data, labels, title="", orientation='vertical'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) if orientation == 'horizontal' else plt.subplots(2, 1, figsize=(5, 12))
    axes[0].pie(data, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[0].set_title(title)
    axes[0].axis('equal')
    if orientation == 'horizontal':
        plt.tight_layout()
    plt.show()

data = [60, 30, 20, 10]
labels = ['A', 'B', 'C', 'D']
plot_pie_subplot(data, labels, title="Gráfico de Sectores de Ejemplo", orientation='vertical')


plt.ion()

class Persona:
    def __init__(self):
        self.nombre = ""
        self.cedula = 0
        self.genero = ""
        self.edad = 0
        self.EEG = Biosenal()
        self.ECG = Biosenal()
        self.datos = ExamenTabular()

    def asignarName(self, nombre):
        self.nombre = nombre

    def asignarCC(self, cedula):
        self.cedula = cedula

    def asignarEdad(self, edad):
        self.edad = edad

    def asignarGenero(self, genero):
        self.genero = genero

    def asignarEEG(self, EEG):
        self.EEG = EEG

    def asignarECG(self, ECG):
        self.ECG = ECG

    def asignarEsta(self, datos):
        self.datos = datos

    def verName(self):
        return self.nombre

    def verCC(self):
        return self.cedula

    def verGenero(self):
        return self.genero

    def verEdad(self):
        return self.edad

    def verEEG(self):
        return self.EEG

    def verECG(self):
        return self.ECG

    def verEsta(self):
        return self.datos


class Sistema:
    def __init__(self):
        self.pacientes = {}

    def ingresarPac(self, paciente):
        self.pacientes[paciente.verCC()] = paciente

    def verPac(self, ced):
        return self.pacientes.get(ced, False)


class Biosenal:
    def __init__(self, data=np.zeros((2, 2))):
        self.data = data
        self.dimension = self.data.ndim
        self.forma = self.data.shape
        self.tamano = self.data.size
        self.canales = self.forma[0]
        self.puntos = self.forma[1]

    def __str__(self):
        return f"""Las características de la matriz de la señal matriz son:
                    forma: {self.forma}
                    dimension: {self.dimension}
                    tamaño: {self.tamano}"""

    def asignarDatos(self, data):
        self.data = data
        self.dimension = self.data.ndim
        self.forma = self.data.shape
        self.tamano = self.data.size
        self.canales = data.forma[0]
        self.puntos = data.forma[1]

    def verForma(self):
        return self.forma

    def verDim(self):
        return self.dimension

    def verTam(self):
        return self.tamano

    def verCanales(self):
        return self.canales

    def verPuntos(self):
        return self.puntos

    def devolver_Canal(self, canal, pmin, pmax):
        if pmin >= pmax:
            return None
        return self.data[canal, pmin:pmax]

    def devolver_Segmento(self, pmin, pmax):
        if pmin >= pmax:
            return None
        return self.data[:, pmin:pmax]

    def promedio(self, eje, pmin, pmax):
        return np.mean(self.data[:, pmin:pmax], axis=eje)

    def desviacion(self, eje, pmin, pmax):
        return np.std(self.data[:, pmin:pmax], axis=eje)


class ExamenTabular(Biosenal):
    def __init__(self, data=np.zeros((2, 2))):
        Biosenal.__init__(self, data)
        self.tablaEx = pd.DataFrame(self.data)
        self.columnas = self.tablaEx.columns

    def verCol(self):
        return self.columnas

    def procesoEstadistico(self):
        return self.tablaEx.describe()

    def devolver_Segmento(self, imin, imax, columnas):
        if imin >= imax:
            return None
        return self.tablaEx.iloc[imin:imax, [columnas]]

    def promedio(self):
        eje = int(input("Ingrese:\n 0- A lo largo de las columnas\n 1- A lo largo de las filas"))
        return self.tablaEx.mean(axis=eje)

    def desviacion(self):
        eje = int(input("Ingrese:\n 0- A lo largo de las columnas\n 1- A lo largo de las filas"))
        return self.tablaEx.std(axis=eje)

    def mediana(self):
        eje = int(input("Ingrese:\n 0- A lo largo de las columnas\n 1- A lo largo de las filas"))
        return self.tablaEx.median(axis=eje)

    def grafico(self, columnas):
        self.tablaEx.boxplot(columnas)


class Graficadora:
    def __init__(self):
        self.fig, self.axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))

    def graficarSenal(self, datos):
        if datos.ndim == 1:
            self.axes[0].plot(datos)
        else:
            for c in range(datos.shape[0]):
                self.axes[0].plot(datos[c, :] + c*600)
        nx = input("Ingrese nombre del eje x: ")
        ny = input("Ingrese nombre del eje y: ")
        tit = input("Titulo: ")
        self.axes[0].set_xlabel(nx)
        self.axes[0].set_ylabel(ny)
        self.axes[0].set_title(tit)
        plt.show()

    def graficaPromedio(self, datos):
        self.axes[1].clear()
        self.axes[1].stem(datos)
        self.axes[1].set_xlabel('Sensores')
        self.axes[1].set_ylabel('Promedio Voltaje (uV)')
        self.axes[1].set_title('Promedios EEG')
        plt.show()

    def graficarHistograma(self, datos):
        self.axes[2].clear()
        self.axes[2].hist(datos)
        self.axes[2].set_xlabel('Bins')
        self.axes[2].set_ylabel('Frecuencia')
        self.axes[2].set_title('Histograma de EEG')
        plt.show()






#11 Para la graficación permitir al usuario elegir entre un diseño de 3subplots (vertical, horizontalo en diagonal), es decir de una columna varias filas o viceversa.

def main():
    sistema = Sistema()
    while True:
        print("\n\t>>>>> MENU PRINCIPAL <<<<<\n1- Ingresar Paciente\n2- Graficar EEG\n3- Estadísticas del Examen\n4- Salir")
        opcion = int(input("--> "))
        if opcion == 1:
            paciente = Persona()
            paciente.asignarName(input("Ingrese nombre: "))
            paciente.asignarCC(int(input("Ingrese cedula: ")))
            paciente.asignarEdad(int(input("Ingrese edad: ")))
            paciente.asignarGenero(input("Ingrese genero: "))
            tipo_archivo = int(input("Indique según archivo a abrir:\n 1-Archivo .mat\n2- Archivo .csv\n--> "))
            if tipo_archivo == 1:
                diccionario = sio.loadmat(input("Ruta del archivo .mat: "))
                llaves = list(diccionario.keys())
                print("Llaves encontradas:")
                for i, llave in enumerate(llaves):
                    print(f"{i+1}: {llave}")
                llave_seleccionada = int(input("Seleccione el número correspondiente a la matriz de interés: "))
                matriz = diccionario[llaves[llave_seleccionada-1]]
                if matriz.ndim == 3:
                    canales, puntos, ensayos = matriz.shape
                    matriz = np.reshape(matriz, (canales, puntos*ensayos), order="F")
                    senal_eeg = Biosenal(matriz)
                    paciente.asignarEEG(senal_eeg)
                else:
                    senal_eeg = Biosenal(matriz)
                    paciente.asignarEEG(senal_eeg)
            else:
                matriz = pd.read_csv(input("Ruta del archivo .csv: "), sep=";")
                examen = ExamenTabular(matriz)
                paciente.asignarEsta(examen)
            sistema.ingresarPac(paciente)

        elif opcion == 2:
            cedula_paciente = int(input("Ingrese la cédula del paciente del que desea hacer la grafica: "))
            paciente = sistema.verPac(cedula_paciente)
            eeg_paciente = paciente.verEEG()
            print(f"Información de la señal EEG: \n {eeg_paciente}")
            while True:
                print("\n¿Qué desea graficar?")
                print("\t1. Todos los canales\n\t2. Un solo canal\n\t3. El promedio\n\t4. Salir de la graficación")
                submenu = int(input("--> "))
                if submenu == 1:
                    min_punto = int(input("Ingrese punto inicial del segmento que desea graficar: "))
                    max_punto = int(input("Ingrese punto final del segmento que desea graficar: "))
                    graficadora = Graficadora()
                    graficadora.graficarSenal(eeg_paciente.devolver_Segmento(min_punto, max_punto))
                elif submenu == 2:
                    canal_seleccionado = int(input("Ingrese el canal que desea graficar: "))
                    min_punto = int(input("Ingrese punto inicial del segmento que desea graficar: "))
                    max_punto = int(input("Ingrese punto final del segmento que desea graficar: "))
                    graficadora = Graficadora()
                    graficadora.graficarSenal(eeg_paciente.devolver_Canal(canal_seleccionado, min_punto, max_punto))
                elif submenu == 3:
                    eje = int(input("Ingrese el eje a lo largo del cual desea hacer el promedio (0 o 1): "))
                    min_punto = int(input("Ingrese punto inicial del segmento que desea graficar: "))
                    max_punto = int(input("Ingrese punto final del segmento que desea graficar: "))
                    graficadora = Graficadora()
                    graficadora.graficaPromedio(eeg_paciente.promedio(eje, min_punto, max_punto))
                elif submenu == 4:
                    break

        elif opcion == 3:
            cedula_paciente = int(input("Ingrese la cédula del paciente del que desea hacer los cálculos: "))
            paciente = sistema.verPac(cedula_paciente)
            examen_tabular = paciente.verEsta()
            if examen_tabular is not None:
                print("Columnas disponibles:")
                print(examen_tabular.verCol())
                columna_seleccionada = input("Ingrese la columna que desea graficar: ")
                examen_tabular.grafico([columna_seleccionada])

        elif opcion == 4:
            break

if __name__ == "__main__":
    main()










