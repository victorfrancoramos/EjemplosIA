## ML_Recomendador - Qué Hace el Código

- **Configuración del Entorno:**  
  - Importa las librerías necesarias de Python y Spark.  
  - Configura el contexto de Spark y ajusta las rutas del sistema para asegurar que las librerías de Spark estén disponibles.

- **Carga y Preparación de Datos:**  
  - Carga los datos de valoraciones de películas desde un archivo de texto, donde cada registro contiene un ID de usuario, un ID de película y una valoración (usando `::` como delimitador).  
  - Parsea los datos en tuplas y filtra las valoraciones con un valor de 0.  
  - Cuenta e imprime el número total de valoraciones, el número de usuarios distintos y el número de películas distintas.

- **División de Datos:**  
  - Divide el conjunto de datos de forma aleatoria en tres subconjuntos:
    - **Conjunto de Entrenamiento (70%)**
    - **Conjunto de Validación (20%)**
    - **Conjunto de Prueba (10%)**

- **Entrenamiento del Modelo y Ajuste de Hiperparámetros:**  
  - Entrena múltiples modelos ALS (Alternating Least Squares) con el subconjunto de entrenamiento utilizando diferentes parámetros (distintos valores de rank y número de iteraciones).  
  - Utiliza el conjunto de validación para:
    - Predecir las valoraciones de las películas.
    - Calcular métricas de rendimiento como el Error Absoluto Medio (MAE) y el coeficiente de determinación R².
  - Selecciona el modelo que presenta el MAE más bajo en la fase de validación.

- **Evaluación sobre el Conjunto de Prueba:**  
  - Vuelve a entrenar el mejor modelo utilizando el conjunto de entrenamiento.  
  - Predice las valoraciones en el conjunto de prueba y evalúa el rendimiento del modelo utilizando las métricas MAE y R².

- **Recomendación de Películas:**  
  - Carga un archivo que contiene las valoraciones de películas del usuario actual, donde una valoración de 0 indica que la película no ha sido valorada aún.  
  - Predice las valoraciones para las películas que el usuario no ha valorado.  
  - Ordena las predicciones y selecciona las 5 recomendaciones principales.  
  - Mapea los IDs de películas con sus respectivos títulos utilizando un archivo separado de consulta y muestra los títulos de las películas recomendadas.

- **Limpieza:**  
  - Detiene el contexto de Spark para liberar recursos del clúster y locales.

> **Nota:** Se ha agregado una versión actualizada llamada `RecomendadorALS_MAE_lambda-0.1_UPDATED.py` que utiliza el nuevo API basado en DataFrames y reemplaza Python 2 por Python 3.
