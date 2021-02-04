#!/usr/bin/env python

# # Ejemplo de modelo de regresión logística
# #  Predicción de mortalidad a partir de bioseñales de una UCI
# #  Valicación Cruzada de 10 Iteraciones
# R.Pingarron <raul.ping4rr0n@gmail.com>
# Si casca es cosa tuya...

import sys
import os
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import *
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

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

## Creamos la sesion de SPARK (Spark2)
spark = SparkSession.builder \
    .master("yarn") \
    .appName("RL_ml_10K-CV") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

# Quitamos el verbosity en la salida de la ejecucion de SPARK
spark.sparkContext.setLogLevel('ERROR')

# Leemos el dataset TRAIN como formato texto (se crea un dataframe):
dataset_train = spark.read.load("/user/raul.pingarron/LR_mortalidad/DatosSinNa_FINAL_nohead_train.txt", format="text")
print "El dataset de entrenamiento tiene {} registros".format(dataset_train.count())

# Como el resultado sigue siendo un dataframe de una unica columna que contiene un string con todo
#  lo pasamos a un RDD para aplicar los maps correnspondientes para tratar la info:
#    Pasamos el dataframe a un RDD mapeandolo como una lista y dividimos la lista en sus elementos separados por espacio:
train_RDD_1 = dataset_train.rdd.map(list).map(lambda lista: lista[0].split())
train_RDD_2 = train_RDD_1.map(lambda lista: (float(lista[0]), Vectors.dense(lista[1:])))
# Convertimos el RDD a un dataframe SQL con dos columnas, label y features:
train_DF = spark.createDataFrame(train_RDD_2,["label","features",])
train_DF.cache()

estimador = LogisticRegression()
MapaParams = ParamGridBuilder() \
    .addGrid(estimador.maxIter, [1, 5, 10, 15]) \
    .addGrid(estimador.regParam, [0.01, 0.03, 0.05, 0.1]) \
    .addGrid(estimador.fitIntercept, [False, True]) \
    .addGrid(estimador.elasticNetParam, [0.0, 0.5, 0.8, 1.0]) \
    .build()
evaluador = BinaryClassificationEvaluator()
cv = CrossValidator(estimator=estimador, estimatorParamMaps=MapaParams, evaluator=evaluador, numFolds=10)

print (" Ejecutando 10-Fold Cross Validation...")
Modelo_CV = cv.fit(train_DF)
MejorModelo = Modelo_CV.bestModel
print ("Los hiper-parametros para el mejor modelo son:")
print ' (regParam): ', MejorModelo._java_obj.getRegParam()
print ' (MaxIter): ', MejorModelo._java_obj.getMaxIter()
print ' (ElasticNetParam): ', MejorModelo._java_obj.getElasticNetParam()
print ' (FitIntercept): ', MejorModelo._java_obj.getFitIntercept()

MejorModelo.write().overwrite().save("/user/raul.pingarron/LR_mortalidad")
print("\n Parametros del modelo salvados en /user/raul.pingarron/LR_mortalidad\n")

# Cargamos el dataset de TEST:
dataset_test = spark.read.load("/user/raul.pingarron/LR_mortalidad/DatosSinNa_FINAL_nohead_test.txt", format="text")
print "El dataset de test tiene {} registros".format(dataset_test.count())
test_RDD_1 = dataset_test.rdd.map(list).map(lambda lista: lista[0].split())
test_RDD_2 = test_RDD_1.map(lambda lista: (float(lista[0]), Vectors.dense(lista[1:])))
test_DF = spark.createDataFrame(train_RDD_2,["label","features",])

# Obtenemos la exactitud de nuestro modelo con respecto al dataset de test:
evaluacion = MejorModelo.evaluate(test_DF)
df_prediccion = evaluacion.predictions
df_predYres = df_prediccion.select('prediction','label')
#eval2 = eval.select('label','prediction')
#eval2.filter(eval2.label == eval2.prediction)
error = df_predYres.filter((df_predYres.label == df_predYres.prediction) & (df_predYres.label == 0.0)).count() / float(df_predYres.count())
print("La exactitud en la prediccion del mejor modelo entrenado es de un %.2f %%"  % (error*100))
ResumenTest = MejorModelo.summary
print ("   Area bajo la curva ROC = %f" % ResumenTest.areaUnderROC)
print ("\n")

#Cerramos la sesion de Spark
SparkSession.stop
