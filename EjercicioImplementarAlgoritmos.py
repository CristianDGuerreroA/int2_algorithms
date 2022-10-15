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
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from itertools import chain, combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



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

def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0] #the total size of the dataset
    total_entr = 0
    
    for c in class_list: #for each class in the label
        total_class_count = train_data[train_data[label] == c].shape[0] #number of the class
        total_class_entr = - (total_class_count/total_row)*np.log2(total_class_count/total_row) #entropy of the class
        total_entr += total_class_entr #adding the class entropy to the total entropy of the dataset
    
    return total_entr

def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0
    
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0] #row count of class c 
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count #probability of the class
            entropy_class = - probability_class * np.log2(probability_class)  #entropy
        entropy += entropy_class
    return entropy

def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique() #unqiue values of the feature
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value] #filtering rows with that feature_value
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list) #calculcating entropy for the feature value
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy #calculating information of the feature value
        
    return calc_total_entropy(train_data, label, class_list) - feature_info #calculating information gain by subtracting

def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label) #finding the feature names in the dataset
                                            #N.B. label is not a feature, so dropping it
    max_info_gain = -1
    max_info_feature = None
    
    for feature in feature_list:  #for each feature in the dataset
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain: #selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = feature
            
    return max_info_feature

def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False) #dictionary of the count of unqiue feature value
    tree = {} #sub tree or node
    
    for feature_value, count in feature_value_count_dict.iteritems():
        feature_value_data = train_data[train_data[feature_name] == feature_value] #dataset with only feature_name = feature_value
        
        assigned_to_node = False #flag for tracking feature_value is pure class or not
        for c in class_list: #for each class
            class_count = feature_value_data[feature_value_data[label] == c].shape[0] #count of class c

            if class_count == count: #count of feature_value = count of class (pure class)
                tree[feature_value] = c #adding node to the tree
                train_data = train_data[train_data[feature_name] != feature_value] #removing rows with feature_value
                assigned_to_node = True
        if not assigned_to_node: #not pure class
            tree[feature_value] = "?" #should extend the node, so the branch is marked with ?
            
    return tree, train_data

def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0: #if dataset becomes enpty after updating
        max_info_feature = find_most_informative_feature(train_data, label, class_list) #most informative feature
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list) #getting tree node and updated dataset
        next_root = None
        
        if prev_feature_value != None: #add to intermediate node of the tree
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else: #add to root of the tree
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
        
        for node, branch in list(next_root.items()): #iterating the tree node
            if branch == "?": #if it is expandable
                feature_value_data = train_data[train_data[max_info_feature] == node] #using the updated dataset
                make_tree(next_root, node, feature_value_data, label, class_list) #recursive call with updated dataset

def id3(train_data_m, label):
    train_data = train_data_m.copy() #getting a copy of the dataset
    tree = {} #tree which will be updated
    class_list = train_data[label].unique() #getting unqiue classes of the label
    make_tree(tree, None, train_data_m, label, class_list) #start calling recursion
    return tree



def predict(tree, instance):
    if not isinstance(tree, dict): #if it is leaf node
        return tree #return the value
    else:
        root_node = next(iter(tree)) #getting first key/feature name of the dictionary
        feature_value = instance[root_node] #value of the feature
        if feature_value in tree[root_node]: #checking the feature value in current tree node
            return predict(tree[root_node][feature_value], instance) #goto next feature
        else:
            return None
        
def evaluate(tree, test_data_m, label):
    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data_m.iterrows(): #for each row in the dataset
        result = predict(tree, test_data_m.iloc[index]) #predict the row
        if result == test_data_m[label].iloc[index]: #predicted value and expected value is same or not
            correct_preditct += 1 #increase correct count
        else:
            wrong_preditct += 1 #increase incorrect count
    accuracy = correct_preditct / (correct_preditct + wrong_preditct) #calculating accuracy
    return accuracy

def id3_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        tree = id3(data_csv, 'Play Tennis')
        #print(tree)
        #Prueba
        test = ['Sunny','Hot','High','Weak']
        accuracy = evaluate(test, data_csv, 'Play Tennis')
        print (accuracy)

########################################################################################################################
def convert_fpg(data_csv):
    dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]
    
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    print(df)
    print(fpgrowth(df, min_support=0.5))
    print(fpgrowth(df, min_support=0.5, use_colnames=True))
    
def fpg_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        convert_fpg(data_csv)
        messagebox.showinfo("", "Que se dice desde el FPGrwoth")   

##########################################################################################################################
def rf_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        #First File
        X = data_csv.iloc[:, 0:4].values
        y = data_csv.iloc[:, 4].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        regressor = RandomForestRegressor(n_estimators=20, random_state=0)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        #Second file
        #X = data_csv.iloc[:, 0:4].values
        #y = data_csv.iloc[:, 4].values
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        #sc = StandardScaler()
        #X_train = sc.fit_transform(X_train)
        #X_test = sc.transform(X_test)
        #regressor = RandomForestRegressor(n_estimators=20, random_state=0)
        #regressor.fit(X_train, y_train)
        #y_pred = regressor.predict(X_test)
        #print(confusion_matrix(y_test,y_pred))
        #print(classification_report(y_test,y_pred))
        #print(accuracy_score(y_test, y_pred))
        


############################################################################################################################
def rl_indicator(data_csv):
    if traverse_matrix_boolean_string(data_csv) == True:
        messagebox.showerror("Error - File", "No se pueden cargar datos String en el DataSet para Regresión Lineal")
    else:
        features = data_csv["age"]
        labels = data_csv["speed"]
        slope_, intercept_, r, p, std_err = stats.linregress(features, labels)

        lineY = list(map(lambda x : slope_ * x + intercept_, features))
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




################################################################################################################
if __name__ == "__main__":
    route_csv = NULL
    data_csv = NULL
    
    route_csv, data_csv = call_csv(route_csv, data_csv)
    view_csv(route_csv, data_csv)
  
    top = Tk() 
    display_gui(top, data_csv)
    
     




