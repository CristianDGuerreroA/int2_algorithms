from asyncio.windows_events import NULL
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
from os import path



def display_gui(top, data_csv):
    rad_btn_var = IntVar()  
    top.geometry("700x100")  
    
    
    top.title('Procesador de algoritmos - Inteligentes 2')  
#    btn_csv_file = Button(top, font=("Verdana", 12),fg='green', activebackground= 'green', highlightbackground='green',text = "Cargar Archivo .csv", command = lambda: call_csv(route_csv,data_csv)).grid(row=0, column=3) 
    Radiobutton(top,font=("Calibri", 13),text="ID3  ", value=1, variable=rad_btn_var, indicatoron= 0, command= lambda: id3_indicator(data_csv)).grid(row=10, column=0)
    Radiobutton(top,font=("Calibri", 13),text="FPGrowth  ",value=2,variable=rad_btn_var, indicatoron= 0, command= lambda: fpg_indicator(data_csv)).grid(row=10, column=1)
    Radiobutton(top,font=("Calibri", 13),text="Random Forest  ", value=3, variable=rad_btn_var, indicatoron= 0, command= lambda: rf_indicator(data_csv)).grid(row=10, column=2)
    Radiobutton(top,font=("Calibri", 13),text="Regresión Lineal  ",value=4,variable=rad_btn_var, indicatoron= 0, command= lambda: rl_indicator(data_csv)).grid(row=10, column=3)
    Radiobutton(top,font=("Calibri", 13),text="Dendograma  ", value=5, variable=rad_btn_var, indicatoron= 0, command= lambda: den_indicator(data_csv)).grid(row=10, column=4)
    Radiobutton(top,font=("Calibri", 13),text="PCA",value=6,variable=rad_btn_var, indicatoron= 0, command= lambda: pca_indicator(data_csv)).grid(row=10, column=5)
#    btn_process_algoritm = Button(top, font=("Verdana", 12),fg='blue', activebackground= 'blue', highlightbackground='blue',text = "Procesar Algoritmo", command= lambda: call_algorithms(rad_btn_var.get())).grid(row=15, column=3) 
    top.mainloop() 
    
def call_csv(route_csv, data_csv):
    name_csv = filedialog.askopenfilename(initialdir="D:\Inteligentes 2\Ejercicio subir nota parcial 1", title='Open File', filetypes=(("CSV files", "*.csv"), ("Python files", "*py"), ("All files", "*.*")))
    route_csv = path.abspath(name_csv)
    data_csv = pd.read_csv(route_csv)
    return route_csv, data_csv
    
    
def view_csv(route_csv, data_csv):
    print('la ruta', route_csv)
    print(data_csv) 

def traverse_matrix_boolean_string (data_csv):
    df = pd.DataFrame(data_csv)
    for key, value in df.itertuples():
        split = value.split(";")
        if verify_is_number(split) == False:
            return True
    return False
            
def verify_is_number(split):
    for item in split:
        val_verify = item.isnumeric()
        if val_verify == False:
            return False

def id3_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        messagebox.showinfo("", "Que se dice desde el ID3")

def fpg_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        messagebox.showinfo("", "Que se dice desde el FPGrowth")

def rf_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        messagebox.showinfo("", "Que se dice desde el Random Forest")































































































from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def lineFunc(x,slope,intercept):
    return slope * x + intercept    

def rl_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        messagebox.showerror("Error - File", "No se pueden cargar datos String en el DataSet para Regresión Lineal")
    else:
        features = data_csv["age"]
        labels = data_csv["speed"]
        slope, intercept, r, p, std_err = stats.linregress(features, labels)

        lineY = list(map(lineFunc, features))
        print (lineY)

        plt.scatter(features, labels)
        plt.plot(features, lineY)
        plt.show()


def Dendograma(dataset):

    X = dataset.iloc[:, [3, 4]].values

    dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

    plt.title('Dendograma')
    plt.xlabel('Clientes')
    plt.ylabel('Distancias Euclidianas')
    plt.show()

def den_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        Dendograma(data_csv)
        messagebox.showinfo("", "Que se dice desde el Dendograma")

def AlgoritmoPCA(dataframe):
    #normalizamos los datos
    scaler=StandardScaler()
    df = dataframe.drop(['comprar'], axis=1) # quito la variable dependiente "Y"
    scaler.fit(df) # calculo la media para poder hacer la transformacion
    X_scaled=scaler.transform(df)# Ahora si, escalo los datos y los normalizo

    #Instanciamos objeto PCA y aplicamos
    pca=PCA(n_components=9) # Otra opción es instanciar pca sólo con dimensiones nuevas hasta obtener un mínimo "explicado" ej.: pca=PCA(.85)
    pca.fit(X_scaled) # obtener los componentes principales
    X_pca=pca.transform(X_scaled) # convertimos nuestros datos con las nuevas dimensiones de PCA

    #Vemos que con 5 componentes tenemos algo mas del 85% de varianza explicada

    #graficamos el acumulado de varianza explicada en las nuevas dimensiones
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    #graficamos en 2 Dimensiones, tomando los 2 primeros componentes principales
    Xax=X_pca[:,0]
    Yax=X_pca[:,1]
    labels=dataframe['comprar'].values
    cdict={0:'red',1:'green'}
    labl={0:'Alquilar',1:'Comprar'}
    marker={0:'*',1:'o'}
    alpha={0:.3, 1:.5}
    fig,ax=plt.subplots(figsize=(7,5))
    fig.patch.set_facecolor('white')
    for l in np.unique(labels):
        ix=np.where(labels==l)
        ax.scatter(Xax[ix],Yax[ix],c=cdict[l],label=labl[l],s=40,marker=marker[l],alpha=alpha[l])

    plt.xlabel("First Principal Component",fontsize=14)
    plt.ylabel("Second Principal Component",fontsize=14)
    plt.legend()
    plt.show()


def pca_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        messagebox.showerror("Error - File", "No se pueden cargar datos String en el DataSet para PCA")
    else:
        AlgoritmoPCA(data_csv)

if __name__ == "__main__":
    route_csv = NULL
    data_csv = NULL
    
    route_csv, data_csv = call_csv(route_csv, data_csv)
    #view_csv(route_csv, data_csv)
  
    top = Tk() 
    display_gui(top, data_csv)
    
     




