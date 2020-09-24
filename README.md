# SIA-TP3: Redes Neuronales
_Perceptron simple y multicapa_

## Pre-requisitos
1. Clonar el repositorio e instalar Python3
2. Para el correcto funcionamiento de este proyecto, se necesita tener instaladas las siguientes librerias externas:
    * [Numpy](https://numpy.org/install/)
    * [Matplotlib](https://matplotlib.org/users/installing.html)

3. Es necesario modificar el archivo de configuracion ``` settings.json ``` de la siguiente manera:

```
{
    "multilayer_lr": 1,
    "multilayer_hidden_nodes_parity": 6,
    "multilayer_max_training_epochs": 5000,
    "multilayer_test_qty": 5,

    "learning_grade": (Float) numero entre 0 y 1 que representa el grado de aprendizaje,
    "operation": ("XOR" | "OR" | "AND") Operador logico a probar,
    "steps": (Integer) Cantidad de iteraciones que el perceptron va a hacer,

    "betha": (Float) Valor de betha. En caso de no especificarse se toma como default 0.5
    "isLinear": (String) ("True" | "False") Determina si el perceptron que se va a utilizar es linear o no. En caso de no especificarse se toma como default verdader
    "function": ("TANH", "LOGISTIC") Determina la funcion a utlizar por el Perceptron no lineal. En caso de no especificarse se toma como default TANH
}
```

## Ejecucion

El proyecto cuenta con diferentes programas para ser ejecutados:

1. Perceptron Simple: Algoritmo del perceptron simple que aprende como funciona los diferentes operadores logicos (XOR/OR/AND) dependendiendo de lo especificado en el archivo de configuracion. Es necesario completar los campos ``` learning_grade, operation, steps``` para el correcto funcionamiento

    Para ejecutar esta opcion correr el comando:
    ```
    python3 SimplePerceptronMain.py
    ```

    Al ejecutar esta opcion el programa implementa un grafico en el que se puede ver las 4 entradas en su clase correspondiente, con su respectivo hiperplano separandolas. Tambien se mostraran por salida estandar las salidas esperadas y las obtenidas por el Perceptron: Esta es la salida, por ejemplo, si se desea probar el operador logico OR

    | File Output | Perceptron Output |
    | ------------- | ------------- |
    | 1 | 1.0|
    | 1 | 1.0|
    | -1 | -1.0|
    | 1 | 1.0|

2. Perceptron Simple Lineal/No Lineal: Algoritmo del perceptron simple lineal y no lineal que aprende en funcion a un set de datos determinado. Al final su ejecucion, se corre un Test, el cual toma de manera aleatoria un conjunto de entradas para entrenar al Perceptron y otra para Testearlo. Es necesario completar los campos ``` learning_grade, operation, steps ```. Se requieren tambien ``` betha, isLinear, function```. En caso de no especificarse estos ultimos se tomaran sus valores defaults.

Para ejecutar esta opcion:
```
python3 SimplePerceptronMain.py
```

Al ejecutar esta opcion, el programa desarrollara un grafico que muestra como fue variando la diferencia entre la salida esperada y la prediccion del Perceptron, asi como tambien una tabla con los resultados del Test ejecutado.

3. Perceptron Multicapa: Perceptron Multicapa que puede aprender tanto el operador logico XOR como discriminar si un numero es par o impar a partir de un conjunto de numeros decimales del 0 al 9 representados por imagenes de 5x7 pixeles. Se deben completar los campos ``` multilayer_lr, multilayer_hidden_nodes_parity,  multilayer_max_training_epochs, multilayer_test_qty```


Para ejecutar la opcion del operador logico:
```
python3 XorResolver.py
```

Para ejecutar la otra opcion:
```
python3 ParityResolver.py
```

Como respuesta, se imprimiran en la terminal, las salidas calculadas con los pesos mas entrenados de cada entrada ingresada.


## Autores
* Ignacio Grasso - [igrasso](https://github.com/igrasso98)
* Bautista Blacker - [bblacker](https://github.com/bautiblacker)
