#!/usr/bin/env python
# coding: utf-8

!git clone https://github.com/Atszvakht/donnees.git

# In[16]:


ls data\datafiles


# In[56]:


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
    StructField("mark",IntegerType(),True),
    StructField("date_de_recolte",DoubleType(),True)
])


# In[58]:


import os

df = None
data_files1 = os.listdir('data\datafiles')
data_files2 = [s for s in data_files1 if "py" not in s]
for data_file in data_files2:
    print(data_file)
    temp_df = spark.read.option("header","false").option("delimiter",";").csv("data/datafiles/"+data_file,schema=schema)
    if df is None :
        df = temp_df
    else :
        df.union(temp_df)
        


# In[59]:


df = df.dropna() 
df.show()


# In[60]:


from pyspark.sql.functions import lit
from pyspark.sql import functions as F


df = df.withColumn('class', 
                     F.when(
                        F.col('mark') == 1, F.lit("rallentisseur")
                     ).otherwise(F.lit('NONrallentisseur')
                   ))
df.count()


# In[61]:


splits = df.randomSplit([0.8,0.2])
df_train = splits[0]
df_test = splits[1]


# In[69]:


from pyspark.ml.feature import StringIndexer,OneHotEncoder
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors

indexer = StringIndexer(inputCol="class",outputCol="label")
vectorAssembler = VectorAssembler(inputCols=["speed","velocity"],outputCol="features")
normalizer = Normalizer(inputCol="features",outputCol="features_norm",p=1.0)


# In[70]:


from pyspark.ml.classification import LogisticRegression


# In[71]:


lr = LogisticRegression(maxIter=10,regParam=0.3,elasticNetParam=0.8)


# In[78]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[indexer,vectorAssembler,normalizer,lr])


# In[80]:


model = pipeline.fit(df_train)


# In[81]:


prediction = model.transform(df_train)


# In[83]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[84]:


eval = MulticlassClassificationEvaluator().setMetricName('accuracy').setLabelCol('label').setPredictionCol('prediction')


# In[85]:


eval.evaluate(prediction)


# In[86]:


model = pipeline.fit(df_test)


# In[87]:


prediction = model.transform(df_test)


# In[88]:


eval.evaluate(prediction)


# In[ ]:




