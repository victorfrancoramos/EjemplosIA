#!/usr/bin/env python

# # Ejemplo de sencillo motor de recomendación de peliculas
# #  Entrenamiento multi-nodo con Apache Spark (HDP distro)
# #  Basado en la MLlib de Spark2 y el algoritmo ALS
# R.Pingarron <raul.ping4rr0n@gmail.com>
# Si casca es cosa tuya...

import sys
from math import sqrt
from operator import add
from os.path import join, isfile
import time
import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.mllib.evaluation import RegressionMetrics

# Cambiamos el encoding a UTF-8 para que no de problemas
reload(sys)
sys.setdefaultencoding('utf8')

# Configuramos el entorno
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = '/usr/hdp/current/spark2-client'
SPARK_HOME = os.environ['SPARK_HOME']
sys.path.insert(0,os.path.join(SPARK_HOME,"python"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib","pyspark.zip"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib","py4j-0.10.4-src.zip"))

# Definimos el contexto de SPARK
conf=SparkConf()
conf.set("spark.executor.memory", "4g")
conf.setAppName("RecomendadorALS_MAE_lambda-0.1")
conf.setMaster('yarn-client')
sc = SparkContext(conf=conf)
# quitamos el verbosity en la salida de la ejecucion de SPARK
sc.setLogLevel("ERROR")

# Cargamos los datos
ratings = sc.textFile("/user/raul.pingarron/recomendador/ratings_all.txt")
# Los parseamos convirtiendolos a un RDD
# Filtro y saco fuera las valoraciones que tienen "0"
ratings_1 = ratings.map(lambda linea: linea.split("::")).map(lambda campo: (int(campo[0]),int(campo[1]),int(campo[2]))).filter(lambda campo: campo[2]!=0)
ratings_RDD = ratings_1.map(lambda campo: (int(campo[0]),int(campo[1]),float(campo[2])))
NumRatings = ratings_RDD.count()
NumUsuarios = ratings_RDD.map(lambda campo: campo[0]).distinct().count()
NumPelis = ratings_RDD.map(lambda campo: campo[1]).distinct().count()
print("\n")
print ("Tenemos %d ratings de %d usuarios para %d peliculas." % (NumRatings, NumUsuarios, NumPelis))

# Dividimos el RDD en tres subconjuntos 70-20-10 para training,validacion y test
training_RDD, validacion_RDD, test_RDD = ratings_RDD.randomSplit([7, 2, 1])
NumTraining = training_RDD.count()
NumValidacion = validacion_RDD.count()
NumTest = test_RDD.count()
print ("Training: %d, Validacion: %d, Test: %d" % (NumTraining, NumValidacion, NumTest))
print ("\n")

# Obtenemos un nuevo RDD a partir del subconjunto de validacion, pero sin los ratings
validacion_sin_ratings = validacion_RDD.map(lambda campo: (campo[0], campo[1]))
# Obtenemos un nuevo RDD a partir del subconjunto de test, pero sin los ratings
test_sin_ratings = test_RDD.map(lambda campo: (campo[0], campo[1]))
# Obtenemos la tupla con los ratings ((usuario,peli) rating) que necesitaremos para sacar el MAE
tupla_ratings = ratings_RDD.map(lambda campo: ((campo[0], campo[1]), campo[2]))

# Entrenamos los modelos con distintos paramentros
#  y evaluamos su MAE con respecto al conjunto de Validacion
mejorRank = 0
mejorNumIter = 0
error_min = float("inf")
R2_min = float("inf")
ranks = [4, 8, 12, 16, 18, 20, 22]
numIters = [5, 10, 15, 20]
for rank in ranks:
   for iteracion in numIters:
      modelo = ALS.train(training_RDD, rank, iteracion, 0.1)
      validacion = modelo.predictAll(validacion_sin_ratings).map(lambda r: ((r[0], r[1]), r[2]))
      prediccionesYratings = validacion.join(tupla_ratings).map(lambda campo: campo[1])
      # Utilizamos la clase RegressionMetrics para comparar el rating actual con el predicho
      metricas = RegressionMetrics(prediccionesYratings)
      MAE_validacion = metricas.meanAbsoluteError
      R2_validacion = metricas.r2
      print ("MAE (validacion) = %f para el modelo entrenado con " % MAE_validacion + "Rank = %d" %rank +", Lambda = 0.1 y %d Iteraciones" % iteracion)
      print (" El coeficiente de determinacion es %f" % R2_validacion)
      if MAE_validacion < error_min:
         error_min = MAE_validacion
         mejorRank = rank
         mejorNumIter = iteracion
         R2_min = R2_validacion
print ("\nLos parametros del mejor modelo obtenido en la fase de entrenamiento son:")
print ("  RANK = %d" % mejorRank)
print ("  ITERACIONES = %d" % mejorNumIter)
print ("  LAMBDA = 0.1")
print ("  Su MAE en la fase de entrenamiento es: %f" % error_min)
print ("    - El coeficiente de determinacion en la fase de entrenamiento es: %f" % R2_min)

# Entrenamos el mejor modelo y evaluamos su MAE con respecto al conjunto de TEST
MejorModelo = ALS.train(training_RDD, mejorRank, mejorNumIter, 0.1)
evaluacion = MejorModelo.predictAll(test_sin_ratings).map(lambda r: ((r[0], r[1]), r[2]))
prediccionesYratings_eval = evaluacion.join(tupla_ratings).map(lambda campo: campo[1])
MetricaFinal = RegressionMetrics(prediccionesYratings_eval)
MAE_test = MetricaFinal.meanAbsoluteError
R2_test = MetricaFinal.r2
print ("\nEl MAE del mejor modelo sobre el cjto test es %f " % (MAE_test))
print ("  - El coeficiente de determinacion sobre el cjto test es %f " % (R2_test))

## RECOMENDACION DE PELICULAS PARA MI USUARIO
# Cargamos el fichero que contiene solo mis ratings para las peliculas:
# De la misma manera se puede hacer para un nuevo usuario, habría que poner sus ratings en un fichero con el mismo formato.
misRatings = sc.textFile("/user/raul.pingarron/recomendador/mis_ratings.txt")
# Los parseamos convirtiendolos a un RDD
misRatings_RDD = misRatings.map(lambda linea: linea.split("::")).map(lambda campo: (int(campo[0]),int(campo[1]),int(campo[2])))
# Parseamos mis ratings quitando aquellas valoraciones=0 (peliculas que no he valorado)
MisPelis_sin_Rating = misRatings_RDD.filter(lambda campo: campo[2]==0).map(lambda campo: (campo[0], campo[1]))
# Recomendamos utilizando el mejor modelo
mi_recomendacion = MejorModelo.predictAll(MisPelis_sin_Rating).map(lambda campo: (campo[1], campo[2]))
# mi_recomendacion.take(3)
# >>> [(3272, 2.0524289409797163), (66, 2.987359474621636), (408, 1.4147540760403898)]

# Tomamos las primeas 5 recomendaciones, ordenadas por orden descedente
mi_recomendacion_ordenada = mi_recomendacion.takeOrdered(5, key = lambda x: -x[1])
peliculas = sc.textFile("/user/raul.pingarron/recomendador/movies.dat").map(lambda linea: linea.split("::")).map(lambda campo: (int(campo[0]),campo[1]))
pelis_dict = dict(peliculas.collect())
print("\n")
print ("Las 5 peliculas recomendadas para mi son:")
for i in [0, 1, 2, 3, 4]:
    print (pelis_dict[mi_recomendacion_ordenada[i][0]])

# Cerramos el contexto de Spark
sc.stop()
