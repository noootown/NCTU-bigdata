from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.feature import StandardScaler
import numpy as np
import sys
from datetime import datetime
from pyspark.mllib.classification import LogisticRegressionWithSGD, SVMWithSGD
from pyspark.mllib.tree import DecisionTree
import math

class DelayRec:
  def __init__(
    self,
    year,
    month,
    dayOfMonth,
    dayOfWeek,
    crsDepTime,
    delay,
    distance,
    cancelled,
    diverted,
  ):
    self.year = year
    self.month = month
    self.dayOfMonth = dayOfMonth
    self.dayOfWeek = dayOfWeek
    self.crsDepTime = crsDepTime
    self.delay = delay
    self.distance = distance
    self.cancelled = cancelled
    self.diverted = diverted
  
  def gen(self):
    v = [
      float(self.delay),
      float(self.month),
      float(self.dayOfMonth),
      float(self.dayOfWeek),
      float(('%04d' % int(self.crsDepTime))[0:2]),
      float(self.distance),
      int(self.diverted),
    ]
    return ('%04d%02d%02d' % (int(self.year), int(self.month), int(self.dayOfMonth)), v)
  
def valid(v, cols):
  for col in cols:
    if v[col] == 'NA':
      return False
  return True

def markDelay(v):
  return LabeledPoint(v[0], np.array(v[1:]))

def prep(sc, file):
  textRDD = sc.textFile(file)
  head = textRDD.first() #the first line of csv
  
  cols = [0, 1, 2, 3, 5, 18, 23, 25]

  return textRDD\
           .filter(lambda line: line != head)\
           .map(lambda x: x.split("\n")[0].split(','))\
           .filter(lambda x: valid(x, cols))\
           .map(lambda x: DelayRec(x[0], x[1], x[2], x[3], x[5], x[25], x[18], x[21], x[23]))\
           .filter(lambda x: x.cancelled == '0')\
           .map(lambda x: (x.gen())[1])\
           .map(markDelay)

def eval_metrics(labelsAndPreds):
    tp = float(labelsAndPreds.filter(lambda r: r[0] == 1 and r[1] == 1).count())
    tn = float(labelsAndPreds.filter(lambda r: r[0] == 0 and r[1] == 0).count())
    fp = float(labelsAndPreds.filter(lambda r: r[0] == 1 and r[1] == 0).count())
    fn = float(labelsAndPreds.filter(lambda r: r[0] == 0 and r[1] == 1).count())

    precision = tp / (tp + fp) if tp + fp > 0 else -1
    recall = tp / (tp + fn) if tp + fn > 0 else -1
    F_measure = 2 * precision * recall / (precision + recall) if precision + recall > 0 else -1
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else -1

    return ([tp, tn, fp, fn], [precision, recall, F_measure, accuracy])

def getScaledData(data):
  features = data.map(lambda x: x.features)
  label = data.map(lambda x: x.label)
  scaler = StandardScaler(withMean = True, withStd = True).fit(features)
  scaled = label\
   .zip(scaler.transform(features.map(lambda x: Vectors.dense(x.toArray()))))\
   .map(lambda x: LabeledPoint(x[0], x[1]))

  return scaled

def getMAE(labelsAndPreds):
  return labelsAndPreds.map(lambda (v, p): math.fabs(v-p)).mean()

def getRMSE(labelsAndPreds):
  return math.sqrt(labelsAndPreds.map(lambda (a, b): (a - b) ** 2).mean())

if __name__ == "__main__":

  sc   = SparkContext()

  train = prep(sc, 'hw4/train')
  t, train_val = train.randomSplit([0.8, 0.2])
  test = prep(sc, 'hw4/test')

  print(train.count())
  print(train_val.count())
  print(test.count())
  
  train_scaled = getScaledData(train)
  train_val_scaled = getScaledData(train_val)
  test_scaled = getScaledData(test)
  
  train.cache()
  train_scaled.cache()
  train_val.cache()
  train_val_scaled.cache()
  test.cache()
  test_scaled.cache()

  iter = 10 ** 4
  step = 10 ** (-5)

  model = LinearRegressionWithSGD.train(train, iter, step) # iter, step size 

  # predict
  predictions_val = model.predict(train_val_scaled.map(lambda x: x.features))
  labelsAndPreds_val = train_val_scaled.map(lambda lp: lp.label).zip(predictions_val).map(lambda (a, b): (b, a))

  predictions = model.predict(test_scaled.map(lambda x: x.features))
  labelsAndPreds = test_scaled.map(lambda lp: lp.label).zip(predictions).map(lambda (a, b): (b, a))
  
  result = open('hw4.txt','a')
  result.write('---------------\n')
  result.write('Validation\n')
  result.write('MAE: %.5f\n' % getMAE(labelsAndPreds_val))
  result.write('RMSE: %.5f\n\n' % getRMSE(labelsAndPreds_val))

  result.write('Test\n')
  result.write('MAE: %.5f\n' % getMAE(labelsAndPreds))
  result.write('RMSE: %.5f\n' % getRMSE(labelsAndPreds))
  result.write('---------------\n')

  '''
  # Build the Decision Tree model
  model_dt = DecisionTree.trainClassifier(train, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=10, maxBins=100)
  
  predictions = model_dt.predict(scaledTestData.map(lambda x: x.features))
  # labelsAndPreds_dt = scaledTestData.map(lambda p: (model_dt.predict(p.features), p.label))
  labelsAndPreds_dt = scaledTestData.map(lambda lp: lp.label).zip(predictions).map(lambda x: (x[1], x[0]))

  m_dt = eval_metrics(labelsAndPreds_dt)[1]

  print("precision = %.2f, recall = %.2f, F1 = %.2f, accuracy = %.2f" % (m_dt[0], m_dt[1], m_dt[2], m_dt[3]))
  '''
