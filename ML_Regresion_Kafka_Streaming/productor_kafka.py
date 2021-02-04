#!/usr/bin/env python

# # Ejemplo de productor KAFKA
# #  Lee las bioseñales de un fichero y las ingesta en el topic de Kafka
# R.Pingarron <raul.ping4rr0n@gmail.com>
# Si casca es cosa tuya...


import sys
import time
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='nodo01-data1:6667,nodo02-data1:6667,nodo03-data1:6667,nodo04-data1:6667', acks='all')

with open('Datos_UCI_test.csv', 'r') as fichero:
    for linea in fichero:
        lista = linea.split()
        lista2 = lista[1:]
        nueva_linea = ' '.join(lista2)
        producer.send('biosignals_UCI', nueva_linea.encode('utf-8'))
        time.sleep(2)
producer.flush()
producer.close()
