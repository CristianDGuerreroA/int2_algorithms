# int2_algorithms
Cómo funciona el algoritmo PCA
El algoritmo PCA es un algoritmo matemático para reducir las dimensiones de conjuntos de datos reteniendo la mayoría de la variación de ellos en vectores llamados componentes principales. Los componentes principales son combinaciones lineales no correlacionadas entre sí de las variables originales y maximizan la varianza de las observaciones. 
Este algoritmo es ampliamente utilizado para identificar patrones en conjuntos de datos con un número dimensiones considerable. 
El primer componente principal captura la mayor cantidad de varianza de los datos, el segundo componente captura la segunda cantidad mayor de la varianza, y así sucesivamente. El número de componentes principales que puede ser obtenido de un conjunto de datos es igual a número de dimensiones que éste posea.
Como datos de entrada tenemos una matriz de datos con filas como observaciones y columnas como variables, la matriz debe contener únicamente variables cuantitativas.
Procedimiento:
-	Calcular la media de cada una de las variables (dimensiones) 
-	Restar la media de cada una de las variables a cada una de las observaciones (restar el vector de medias a cada una de las filas de la matriz) 
-	Calcular la matriz de covarianza 
-	Calcular los eigenvectores y eigenvalores de la matriz de covarianza 
-	Graficar componentes principales
Como datos de salida tenemos los eigenvectores (componentes principales) y eigenvalores

¿Qué es un eigenvector?
En álgebra lineal, los vectores propios, eigenvectores o autovectores de un operador lineal son los vectores no nulos que, cuando son transformados por el operador, dan lugar a un múltiplo escalar de sí mismos, con lo que no cambian su dirección. Esta escalar lambda recibe el nombre valor propio, autovalor o valor característico. A menudo, una transformación queda completamente determinada por sus vectores propios y valores propios.

José Alquicira. (2016). Análisis de componentes principales (PCA). 2022, Octubre 16, Conogasi.org Sitio web: https://conogasi.org/articulos/analisis-de-componentes-principales-pca/

