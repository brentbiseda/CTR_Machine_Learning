#!/usr/bin/env python

import re
import ast
import time
import numpy as np
import pandas as pd	
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
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

from pyspark.ml.classification import RandomForestClassifier
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

testDF = sqlContext.read.parquet("gs://"+BUCKET+"/test.parquet") # This loads a Data Frame
trainDF = sqlContext.read.parquet("gs://"+BUCKET+"/train.parquet")

HEADER = trainDF.columns
INTEGER_FEATURES = HEADER[0:14] # These are the integer features
CATEGORICAL_FEATURES = HEADER[14:] # These are the categorical features

for feature in INTEGER_FEATURES[1:]:
    trainDF = trainDF.withColumn(feature, trainDF[feature].cast(StringType()))
    trainDF = trainDF.fillna({feature:''})

CATEGORICAL_FEATURES = INTEGER_FEATURES + CATEGORICAL_FEATURES

# We will track which features to eliminate from our dataframe
featuresToDrop = set()
featuresToDrop.add('x11')
featuresToDrop.add('x13')
featuresToDrop.add('x14')
featuresToDrop.add('x18')
featuresToDrop.add('x21')

def keepTopK(df, dftest, K, categoricalColumnstoImpute):
    for col in categoricalColumnstoImpute:
        mostCommon = df.select(col).groupby(col).count()\
                            .orderBy('count', ascending=False) \
                            .limit(K).collect()
            
        mostCommonSet = set([x[0] for x in mostCommon])
               
        df = df.withColumn(col, F.when(~df[col].isin(mostCommonSet), "RECODED").otherwise(df[col]))
        
        dftest = dftest.withColumn(col, F.when(~dftest[col].isin(mostCommonSet), "RECODED") \
                        .otherwise(dftest[col]))
    
    print("Successfully Recoded Top K Categorical Values")
    
    return (df, dftest)

trainDF, testDF = keepTopK(trainDF, testDF, 10000, ['x5', 'x16', 'x17', 'x25', 'x29', 'x34', 'x37']) # Select 10,000 top categories

distinct = []
for col in CATEGORICAL_FEATURES:
    distinct.append(set(trainDF.select(col).distinct().rdd.map(lambda x: x[0]).collect()))

distinctDict = dict((k, v) for k, v  in zip(CATEGORICAL_FEATURES, distinct))

def imputeValues(df, dftest):
    categoricalColumnstoImpute = CATEGORICAL_FEATURES[1:]
    
    # Impute categorical features
    for col in categoricalColumnstoImpute:
        mostCommon = df.select(col).groupby(col).count()\
                            .orderBy('count', ascending=False) \
                            .limit(1).collect()[0][0]
        if mostCommon == "":
            mostCommon = "EMPTY"
        
        print(f"Column {col} has most common {mostCommon}")
        
        df = df.withColumn(col, F.when((df[col].isNull() | (df[col] == '')), mostCommon) \
                                .otherwise(df[col]))
        
        dftest = dftest.withColumn(col, F.when((dftest[col].isNull() | (dftest[col] == '') | (~dftest[col].isin(distinctDict[col]))), mostCommon) \
                        .otherwise(dftest[col]))
    print("Successfully Imputed Categorical Values")
    
    # Assure there is no missing values
    for col in categoricalColumnstoImpute:
        assert df.filter(df[col].isNull()).count() == 0, f"Column {col} contains NULL value(s)"
        assert df.filter(df[col] == '').count() == 0, f"Column {col} contains empty string(s)"
    
        assert dftest.filter(dftest[col].isNull()).count() == 0, f"Column {col} contains NULL value(s)"
        assert dftest.filter(dftest[col] == '').count() == 0, f"Column {col} contains empty string(s)"
    
    print("Successfully Imputed All Values and Passed Tests")
    return (df, dftest)

trainDF, testDF = imputeValues(trainDF, testDF)

trainDF.write.parquet("gs://"+BUCKET+"/trainImputed.parquet")
testDF.write.parquet("gs://"+BUCKET+"/testImputed.parquet")

print("complete")
