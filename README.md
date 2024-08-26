## Portafolio de Implementación

Para el primer portafolio, decidí hacer uso del modelo de **Regresión Lineal Múltiple**. Este modelo se encargará de predecir la cantidad de calorías en un alimento en base a la información nutrimental de este. El dataset se compone de 5 variables: Carbohidratos, Lípidos, Proteína, Sodio, Calorías.

### Creación del Modelo

Después de dividir el dataset en los conjuntos de **training** y **test**, se entrena el modelo utilizando la función `train()`. Antes de esto, es necesario definir algunas variables:

- `w` se convierte en un vector en vez de un valor escalar debido a que contamos con múltiples features para asignar peso. Este parámetro se inicializa como un vector de 1s.
- `b` se inicializa en 1.
- Los hiperparámetros de **epochs** y **alpha** (learning rate) nos ayudan a ajustar el modelo utilizando gradient descent.

La función `train()` actualizará los parámetros `w` y `b` llamando a la función `update_w_and_b()` para cada epoch. Es aquí donde se calculan los gradientes, sacando las derivadas de la función de costo con respecto a ambas variables (`W` y `b`). El valor de los gradientes multiplicado por `alpha` (learning rate) nos da el tamaño del salto. Si restamos este tamaño del salto a los parámetros de la pasada iteración, obtendremos los nuevos parámetros `W` y `b`. Cabe aclarar que para las operaciones de gradientes se tuvo que utilizar el producto punto debido a los múltiples pesos asignados a los features. Al final, tendremos los parámetros `w` y `b` optimizados para comenzar a hacer predicciones.

### Experimentación

Al principio, se entrenó el modelo con 4 features: Carbohidratos, Lípidos, Proteína, Sodio. Sin embargo, al observar que existía mucho error, se quitó Sodio de los features seleccionados. De esta manera, los resultados mejoraron, pero aún así no eran los mejores. También se ajustaron hiperparámetros como `alpha` para tratar de obtener mejores resultados.

### Resultados

El MSE del modelo fue: `1701.7691734179903`. Este fue un valor alto, demostrando que el modelo no es muy bueno. Esto pudo ser causado por una mala calidad del dataset. Este dataset fue creado por mí y contiene más de 400 registros únicos de alimentos. Sin embargo, no demuestra la información nutrimental completa, posiblemente omitiendo variables que pueden tener una gran influencia en el número de calorías.

### Reflexión y Aprendizaje

Mi modelo no obtuvo los mejores resultados, esto debido a la falta de un dataset más extenso con la capacidad de realizar **Feature Engineering**. Sin embargo, pude entender cómo es posible asignar pesos a múltiples features y el uso del algoritmo de gradient descent para optimizar estos pesos en el modelo. Aún así, buscaré un mejor dataset para la siguiente entrega.
