#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!git clone https://github.com/Atszvakht/donnees.git


# In[3]:


ls donnees\datafiles


# In[1]:


from pyspark.sql.types import StructType,StructField,DoubleType,IntegerType,BooleanType


schema = StructType([
    StructField("index",IntegerType(),True),
    StructField("laltitude",DoubleType(),True),
    StructField("longitude",DoubleType(),True),
    StructField("speed",DoubleType(),True),
    StructField("speedAccuracy",DoubleType(),True),
    StructField("direction",DoubleType(),True),
    StructField("directionAccuracy",DoubleType(),True),
    StructField("velocity",DoubleType(),True),
    StructField("x",DoubleType(),True),
    StructField("y",DoubleType(),True),
    StructField("z",DoubleType(),True),
    StructField("label",IntegerType(),True),
    StructField("date_de_recolte",DoubleType(),True)
])


# In[49]:


import os
"""from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)"""

df = None
data_files1 = os.listdir('donnees\datafiles')
data_files2 = [s for s in data_files1 if "py" not in s]
for data_file in data_files2:
    print(data_file)
    temp_df = spark.read.option("header","false").option("delimiter",";").csv("donnees/datafiles/"+data_file,schema=schema)
    if df is None :
        df = temp_df
    else :
        df = df.union(temp_df)
df.count()
        


# In[50]:


from pyspark.sql.functions import lit
from pyspark.sql import functions as F
import pandas


df = df.withColumn('class', 
                     F.when(
                        F.col('label') == 1, F.lit("rallentisseur")
                     ).otherwise(F.lit('NONrallentisseur')
                   ))
df = df.dropna()
df.count()


# In[51]:


splits = df.randomSplit([0.5,0.5])
df_train = splits[0]
df_test = splits[1]


# In[52]:


from pyspark.ml.feature import StringIndexer,OneHotEncoder
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors

indexer = StringIndexer(inputCol="class",outputCol="newClass")
vectorAssembler = VectorAssembler(inputCols=["speed","velocity"],outputCol="features")
normalizer = Normalizer(inputCol="features",outputCol="features_norm",p=1.0)


# In[53]:


from pyspark.ml.classification import LogisticRegression


# In[54]:


lr = LogisticRegression(maxIter=10,regParam=0.3,elasticNetParam=0.8)


# In[55]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[indexer,vectorAssembler,normalizer,lr])


# In[56]:


model = pipeline.fit(df_train)


# In[57]:


prediction = model.transform(df_train)


# In[58]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[59]:


eval = MulticlassClassificationEvaluator().setMetricName('accuracy').setLabelCol('label').setPredictionCol('prediction')


# In[60]:


eval.evaluate(prediction)


# In[61]:


"""df_test_No_Mark = df_test.withColumn("label",lit(None))
df_test_No_Mark = df_test_No_Mark.withColumn('class', F.lit(None))
model = pipeline.fit(df_test_No_Mark)
prediction = model.transform(df_test_No_Mark)"""


# In[62]:


eval.evaluate(prediction)


# In[63]:


model = pipeline.fit(df_test)


# In[64]:


prediction = model.transform(df_test)


# In[67]:


prediction.filter(prediction.prediction ==1).collect()


# In[101]:


eval = MulticlassClassificationEvaluator().setMetricName('accuracy').setLabelCol('label').setPredictionCol('prediction')


# In[102]:


eval.evaluate(prediction)

