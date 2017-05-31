from pyspark import SparkConf, SparkContext
from operator import add
import sys
from datetime import datetime

def main(sc, filename):
  a = datetime.now()
  textRDD = sc.textFile(filename)
  words = textRDD.flatMap(lambda x: x.split(' ')).map(lambda x: (x, 1))
  wordcount = words.reduceByKey(add).collect()

  arr = {}
  for wc in wordcount:
    arr[wc[0]] = wc[1]

  # sort and print
  sort = sorted(arr.items(), key=lambda item: item[1])[::-1][:20]

  for word in sort:
    print(word)

  b = datetime.now()
  print(b-a)

if __name__ == "__main__":

  # Configure Spark
  conf = SparkConf().setAppName("Count word")
  conf = conf.setMaster("local[*]")
  sc   = SparkContext(conf=conf)
  filename = sys.argv[1]
  # Execute Main functionality
  main(sc, filename)
