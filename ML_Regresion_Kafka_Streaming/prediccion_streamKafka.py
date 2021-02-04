#!/usr/bin/env python

# # Ejemplo de aplicación de un modelo de predicción de mortalidad basado en
# #  regresión logística aplicado en STREAMING (SPARK) sobre datos ingestados
# #  en tiempo real sobre una cola persistente bajo Apache KAFKA
# R.Pingarron <raul.ping4rr0n@gmail.com>
# Si casca es cosa tuya...

import sys
import os
import time
from pyspark.sql.functions import split
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT

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
    .appName("streaming_prediction") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

# Quitamos el verbosity en la salida de la ejecucion de SPARK
spark.sparkContext.setLogLevel('ERROR')

#Cargamos el mejor modelo que hemos salvado:
MejorModelo = LogisticRegressionModel.load("/user/raul.pingarron/LR_mortalidad")

# Creamos el Streaming DataFrame :
df = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "nodo01-data1:6667,nodo02-data1:6667,nodo03-data1:6667,nodo04-data1:6667") \
  .option("subscribe", "biosignals_UCI") \
  .option("startingOffsets", "earliest") \
  .load()

cols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]

# Aplicamos las transformaciones necesarias sobre el Streaming DataFrame :
stream_df = df.selectExpr("CAST(value AS STRING)")
split_col = split(stream_df['value'], ' ')
for i in range(21): \
   stream_df = stream_df.withColumn(str(i), split_col.getItem(i))

stream_df1 = stream_df.select(cols)
stream_df2 = stream_df1.select(*(col(c).cast("float").alias(c) for c in stream_df1.columns))
vecAssembler = VectorAssembler(inputCols=stream_df2.columns, outputCol="features")
stream_df3 = vecAssembler.transform(stream_df2)
stream_df_test = stream_df3.select("features")

prediction = MejorModelo.transform(stream_df_test)
prediccion = prediction.select("features", "probability", "prediction")
print (prediccion)

query = prediccion \
    .writeStream \
    .outputMode('append') \
    .format("console") \
    .trigger(processingTime='2 seconds') \
    .start()
# Paramos la computacion en streaming despues de 15 segundos
time.sleep(240)
query.stop()

#Cerramos la sesion de Spark
SparkSession.stop
