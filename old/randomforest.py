#!/usr/bin/env python

# imports
import re
import ast
import time
import numpy as np
import pandas as pd
# import seaborn as sns
# import networkx as nx
# import matplotlib.pyplot as plt
import pyspark
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import sys
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import StandardScaler
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoderEstimator, OneHotEncoderModel
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StandardScaler, Imputer, VectorAssembler, SQLTransformer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



conf = pyspark.SparkConf().setAll([ ('spark.executor.pyspark.memory', '11g'), ('spark.driver.memory','11g')])
sc = pyspark.SparkContext(conf=conf)
# sc = pyspark.SparkContext()


############## YOUR BUCKET HERE ###############

BUCKET="w261-finalproject"

############## (END) YOUR BUCKET ###############

sqlContext = SQLContext(sc)

# wikiRDD = sc.textFile("gs://"+BUCKET+"/trainImputed.parquet")

testDF = sqlContext.read.parquet("gs://"+BUCKET+"/testImputed.parquet") # This loads a Data Frame
trainDF = sqlContext.read.parquet("gs://"+BUCKET+"/trainImputed.parquet")

HEADER = trainDF.columns
CATEGORICAL_FEATURES = HEADER[1:] 

featuresToDrop = set()
featuresToDrop.add('x11')
featuresToDrop.add('x13')
featuresToDrop.add('x14')
featuresToDrop.add('x18')
featuresToDrop.add('x21')

trainDF = trainDF.select([c for c in trainDF.columns if c not in featuresToDrop])
testDF = testDF.select([c for c in testDF.columns if c not in featuresToDrop])

cols_in = ['x1','x2', 'x3','x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x12',
        'x15', 'x16', 'x17', 'x19', 'x20', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27',
        'x28', 'x29', 'x30', 'x31', 'x32', 'x33', 'x34', 'x35', 'x36', 'x37', 'x38', 'x39']

cols_out = ['x1_OHE','x2_OHE', 'x3_OHE','x4_OHE', 'x5_OHE', 'x6_OHE', 'x7_OHE', 'x8_OHE', 'x9_OHE', 'x10_OHE', 'x12_OHE',
        'x15_OHE', 'x16_OHE', 'x17_OHE', 'x19_OHE', 'x20_OHE', 'x22_OHE', 'x23_OHE', 'x24_OHE', 'x25_OHE', 'x26_OHE', 'x27_OHE',
        'x28_OHE', 'x29_OHE', 'x30_OHE', 'x31_OHE', 'x32_OHE', 'x33_OHE', 'x34_OHE', 'x35_OHE', 'x36_OHE', 'x37_OHE', 'x38_OHE', 'x39_OHE']

# String Indexing for categorical features
indexers = [StringIndexer(inputCol=x, outputCol=x+"_tmp") for x in cols_in]

# One-hot encoding for categorical features
encoders = [OneHotEncoder(dropLast=False, inputCol=x+"_tmp", outputCol=y) for x,y in zip(cols_in, cols_out)]
    
tmp = [[i,j] for i,j in zip(indexers, encoders)]
tmp = [i for sublist in tmp for i in sublist]

assembler = VectorAssembler(inputCols = cols_out, outputCol = "features")

labelIndexer = StringIndexer(inputCol="y", outputCol="label")

tmp += [assembler,labelIndexer]

pipeline = Pipeline(stages=tmp)

allData = pipeline.fit(trainDF).transform(trainDF)

allData.cache()

trainData, validData = allData.randomSplit([0.8,0.2], seed=1)

randforest = RF(labelCol="label", featuresCol="features", numTrees = 100)

rf_fit = randforest.fit(trainData)

transformed = rf_fit.transform(validData)

results = transformed.select(["probability", "label"])

results_collect = results.collect()

results_list = [(float(i[0][0]),1.0-float(i[1])) for i in results_collect]

score = sc.parallelize(results_list)

metrics = metric(score)

print("The ROC score is (numTrees=100): ", metrics.areaUnderROC)

print("complete")
