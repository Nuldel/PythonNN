# Red Neuronal con Python
## Descripción
Una red neuronal simple con autoencoder (opcional) y parámetros configurables, adaptada al reconocimiento de imágenes de números (base de datos MNIST de números escritos a mano). Tiene una etapa de aprendizaje autómatico y una testeo (con un set separado de imágenes). La clase:
```
GeneralNN(sizes, actFunctions, dropOut=0.0)
```
construye la red, donde **sizes** es un arreglo de tamaños de cada capa (número de neuronas), **actFunctions** es un arreglo del mismo largo con la función de activación de cada capa, y dropOut es la probabilidad de dropeo aleatorio de una neurona.

Esta es una generalización de la red. Para aplicarla a las imágenes, se usa particularmente la función:
```
buildAndTrainImages(n_epochs=5, batch_size_train=64, batch_size_test=1000, learning_rate=0.01, log_interval=10, encoder_epochs=0)
```
que contruye la red, la entrena y la prueba, todo de manera automática. **n-epochs** es la cantidad de *épocas* de aprendizaje/testeo, cada una con una cantidad de imágenes dada en **batch_size_train** y **batch_size_test** respectivamente; hay un *learning rate* configurable, así como el intervalo de reporte en pantalla del error actual, y **encoder_epochs** es un número opcional de épocas previas al entrenamiento real en el cual se usa un autoencoder (= 0 para no usarlo).

La idea es aproximar el error de aprendizaje en cada época y luego reportar los errores de intentar clasificar las imágenes de un set que no se usó en ese aprendizaje. Idealmente, se podría contruir una red y entrenarla lo suficiente para decidir qué número del 0 al 9 fue dibujado por un humano.

Para más detalles del análisis teórico realizado, ver **README.pdf**.
