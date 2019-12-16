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

BUCKET=" "

############## (END) YOUR BUCKET ###############

sqlContext = SQLContext(sc)

# testDF = sqlContext.read.parquet("gs://"+BUCKET+"/test.parquet") 
# trainDF = sqlContext.read.parquet("gs://"+BUCKET+"/train.parquet")

