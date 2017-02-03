# Convolutional Neural Network + TensorFlow

@Autor: Filiberto Vilca Apaza

## Resumen
El presente trabajo se desarrolló con el fin de comprender y analizar las diferentes funcionalidad de tensorflow a fin de levantar un red neuronal convolucional en TensorFlow, este informe describe el dataset utilizado, las herramientas utilizadas, una descripción de los archivos generados en python, la configuración inicial de  la red neuronal(Capas, Modelo, función de activación), los requerimientos del sistema y los resultados obtenidos. El siguient trabajo esta basado en https://www.tensorflow.org/tutorials/deep_cnn/

##DataSet
CIFAR-10,  es un problema de referencia común en el aprendizaje automático. El problema es clasificar las imágenes RGB de 32 x 32 píxeles en 10 categorías: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.


## Objetivo:
Construir una red neuronal convolucional relativamente pequeña (CNN) para reconocer imágenes. En el proceso, este tutorial:

## Herramientas  y Librerías usadas:
- numpy: calculos cientificos para python.
- six: Paquete permite compatibilidad con Python 2 y Python 3 en el código
- TensorBoard: Este visualizador facilita la comprensión del flujo de los tensores, depuración y optimización. TensorFlow, traza métricas cuantitativas sobre la ejecución de su programa de forma gráfica
-- Para ejecutar TensorBoard, escribir en la línea de comandos:
-- tensorboard --logdir=path/to/log-directory  --port 6006
  

## Configuración.
Para obtener los resultados finales se utilizó el siguiente Hardware:
- **Windows 10**
- Intel Core **i7 @4GhZ**
- CUDA Graphic card (**NVIDIA GTX750 Ti**)
- tamaño del lote (Batch Size)
- INITIAL_LEARNING_RATE = 0.1 
LEARNING_RATE_DECAY_FACTOR = 0.1 


|       File       	|                      Descripción                     	|
|:----------------:	|:----------------------------------------------------:	|
| cifar10.py       	| Crea el modelo CIFAR-10.                             	|
| cifar10_input.py 	| Lee el formato de archivo binario nativo CIFAR-10.   	|
| cifar10_train.py 	| Entrena un modelo CIFAR-10 en una CPU o GPU.         	|
| cifar10_eval.py  	| Evalúa el desempeño predictivo de el modelo CIFAR-10 	|


##cifar10.py – builds the model

### Flags por defecto.
- data_dir (ruta al directorio de datos CIFAR-10): 'C:/tmp/cifar10_data'
- batch_size (número de imágenes a procesar en un lote): 128
- use_fp16 (entrenar el modelo usando fp16): False

###constantes Globales Dataset:
-IMAGE_SIZE: 24 
-NUM_CLASSES: 10 
-NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN: 50000 
-NUM_EXAMPLES_PER_EPOCH_FOR_EVAL: 10000 

-Constantes que describen el proceso de entrenamiento: 
-MOVING_AVERAGE_DECAY: 0.9999 
-NUM_EPOCHS_PER_DECAY: 350.0 
-LEARNING_RATE_DECAY_FACTOR: 0.1 
-INITIAL_LEARNING_RATE: 0.1 


###This CIFAR-10 model consist from the following layers:

- conv1 – 2D convolución and RELU
- pool1 – pooling 
- norm1 – Normalización
- conv2 (igual que conv1)
- norm2 (igual que norm1)
- pool2 (igual que pool1)
- local3 – capa completamente conectada con funcion de activacion Relu (local 4)
- softmax_linear – clasificador Softmax.

##cifar10_train.py – Training the TensorFlow CIFAR-10 model

###flags por defecto:

- train_dir (directorio donde escribir los registros de eventos y el punto de control): 'C:/tmp/cifar10_train'
- max_steps (número de lotes a ejecutar): 10000
- log_device_placement (si registra la colocación del dispositivo): true ('log_device_placement' ayuda a averiguar qué dispositivos están asignados a sus operaciones y tensores, usados en Computación Multi GPU en TensorFlow).

##cifar10_eval.py – evaluating the performance of TensorFlow CIFAR-10 model

###default flags:

- eval_dir (directorio donde se guarda los eventos-logs): 'C:/tmp/cifar10_eval'
- eval_data (al igual que  'test o train_eval')
- checkpoint_dir (donde leer los checkpoints del modelo): '/tmp/cifar10_train'
- eval_interval_secs (Con qué frecuencia ejecuta la evaluación) - cada 5 minutos (60*5)
- num_examples (numero de ejemplos para evaluar) – 10’000
- run_once (si es *true* ejecutará una sola vez) – false

 ## Resultados
- pérdida inicial  razonable 4.68, se continuo porque así aseguramos que puede lograr la precisión del entrenamiento en un 100% y en una pequeña porción de los datos de entrenamiento.
- se obtuvo una precisión de 81%  con 10000 pasos en 30 min

## Otras Notas:
- se realizó una prueba con un millón de pasos , pero las condiciones de corte de luz no lo  permitieron