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

def rl_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        messagebox.showerror("Error - File", "No se pueden cargar datos String en el DataSet para Regresión Lineal")

def den_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        messagebox.showinfo("", "Que se dice desde el Dendograma")


def pca_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        messagebox.showinfo("", "Que se dice desde el Random Forest")

if __name__ == "__main__":
    route_csv = NULL
    data_csv = NULL
    
    route_csv, data_csv = call_csv(route_csv, data_csv)
    view_csv(route_csv, data_csv)
  
    top = Tk() 
    display_gui(top, data_csv)
    
     




