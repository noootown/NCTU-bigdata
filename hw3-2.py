
from pyspark import SparkConf, SparkContext
from operator import add
import sys
from datetime import datetime

def fetch(arr, index):
  try:
    return int(arr[0].split(',')[index])
  except:
    return 0

def main(sc, filename):
  textRDD = sc.textFile(filename)
  passen = textRDD.map(lambda x: x.split("\n")).map(lambda x: (fetch(x, 11), fetch(x, 3)))
  passenValue = passen.reduceByKey(add).collect()
  passenCount = passen.map(lambda (a, b): (a, 1 if b > 0 else 0)).reduceByKey(add).collect()

  value = {}
  count = {}

  for wc in passenValue:
    value[wc[0]] = wc[1]

  for wc in passenCount:
    count[wc[0]] = wc[1]

  return value, count

if __name__ == "__main__":

  # Configure Spark
  conf = SparkConf().setAppName("Count word")
  sc   = SparkContext(conf=conf)

  a = datetime.now()

  v, c = main(sc, 'yellow_tripdata_2015-01.csv')

  # Execute Main functionality
  
  for i in range(2, 13, 1):
    print('round', i)
    d1, d2 = main(sc, 'yellow_tripdata_2015-%02d.csv' % i)
    for key, value in d1.items():
      v[key] += value
    for key, value in d2.items():
      c[key] += value
  
  # total value
  print(v)
  # total count
  print(c)
  
  for key, value in v.items():
    if c[key] != 0 and value != 0 and key != 0:
      print("%s: %.5f" % (key, float(value) / float(c[key])))

  b = datetime.now()
  print(b-a)
