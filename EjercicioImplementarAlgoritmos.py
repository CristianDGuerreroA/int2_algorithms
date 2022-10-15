from asyncio.windows_events import NULL
from distutils.command.config import config
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
from os import path
import numpy as np
import math
import copy
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


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
    name_csv = filedialog.askopenfilename(initialdir="D:\Inteligentes 2\int2_algorithms", title='Open File', filetypes=(("CSV files", "*.csv"), ("Python files", "*py"), ("All files", "*.*")))
    route_csv = path.abspath(name_csv)
    data_csv = pd.read_csv(route_csv)
    return route_csv, data_csv
    
    
def view_csv(route_csv, data_csv):
    print('la ruta', route_csv)
    print(data_csv)

def traverse_matrix_boolean_string (data_csv):
    df = pd.DataFrame(data_csv)
    for item, row in df.iterrows():
        values = row.values.tolist()
        if verify_is_number(values) == False:
            return True
    return False
            
def verify_is_number(values):
    for item in values:
        val_com = str(item).isnumeric()
        if val_com == False:
            return False

class Node(object):
    def __init__(self):
        self.value = None
        self.decision = None
        self.childs = None


def findEntropy(data, rows):
    yes = 0
    no = 0
    ans = -1
    idx = len(data[0]) - 1
    entropy = 0
    for i in rows:
        if data[i][idx] == 'Yes':
            yes = yes + 1
        else:
            no = no + 1

    x = yes/(yes+no)
    y = no/(yes+no)
    if x != 0 and y != 0:
        entropy = -1 * (x*math.log2(x) + y*math.log2(y))
    if x == 1:
        ans = 1
    if y == 1:
        ans = 0
    return entropy, ans


def findMaxGain(data, rows, columns):
    maxGain = 0
    retidx = -1
    entropy, ans = findEntropy(data, rows)
    if entropy == 0:
        """if ans == 1:
            print("Yes")
        else:
            print("No")"""
        return maxGain, retidx, ans

    for j in columns:
        mydict = {}
        idx = j
        for i in rows:
            key = data[i][idx]
            if key not in mydict:
                mydict[key] = 1
            else:
                mydict[key] = mydict[key] + 1
        gain = entropy

        # print(mydict)
        for key in mydict:
            yes = 0
            no = 0
            for k in rows:
                if data[k][j] == key:
                    if data[k][-1] == 'Yes':
                        yes = yes + 1
                    else:
                        no = no + 1
            # print(yes, no)
            x = yes/(yes+no)
            y = no/(yes+no)
            # print(x, y)
            if x != 0 and y != 0:
                gain += (mydict[key] * (x*math.log2(x) + y*math.log2(y)))/14
        # print(gain)
        if gain > maxGain:
            # print("hello")
            maxGain = gain
            retidx = j

    return maxGain, retidx, ans


def buildTree(data, rows, columns, attribute, X):

    maxGain, idx, ans = findMaxGain(X, rows, columns)
    root = Node()
    root.childs = []
    # print(maxGain
    #
    # )
    if maxGain == 0:
        if ans == 1:
            root.value = 'Yes'
        else:
            root.value = 'No'
        return root

    root.value = attribute[idx]
    mydict = {}
    for i in rows:
        key = data[i][idx]
        if key not in mydict:
            mydict[key] = 1
        else:
            mydict[key] += 1

    newcolumns = copy.deepcopy(columns)
    newcolumns.remove(idx)
    for key in mydict:
        newrows = []
        for i in rows:
            if data[i][idx] == key:
                newrows.append(i)
        # print(newrows)
        temp = buildTree(data, newrows, newcolumns)
        temp.decision = key
        root.childs.append(temp)
    return root


def traverse(root):
    print(root.decision)
    print(root.value)

    n = len(root.childs)
    if n > 0:
        for i in range(0, n):
            traverse(root.childs[i])


def calculate(attribute, X, data_csv):
    rows = [i for i in range(0, 14)]
    columns = [i for i in range(0, 4)]
    root = buildTree(X, rows, columns, attribute)
    root.decision = 'Start'
    traverse(root)


def id3_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        X = data_csv.iloc[:, 1:].values
        attribute = ['outlook', 'temp', 'humidity', 'wind']
        calculate(attribute, X)

def fpg_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        messagebox.showinfo("", "Que se dice desde el FPGrowth")

def rf_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        messagebox.showinfo("", "Que se dice desde el Random Forest")




































































































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

def pca_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        messagebox.showerror("Error - File", "No se pueden cargar datos String en el DataSet para PCA")

if __name__ == "__main__":
    route_csv = NULL
    data_csv = NULL
    
    route_csv, data_csv = call_csv(route_csv, data_csv)
    view_csv(route_csv, data_csv)
  
    top = Tk() 
    display_gui(top, data_csv)
    
     




