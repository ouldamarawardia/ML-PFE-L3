#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
df = df.dropna()
df.count()


# In[4]:


df_test = None
data_files1 = os.listdir('donnees\Test')
data_files2 = [s for s in data_files1 if "py" not in s]
for data_file in data_files2:
    print(data_file)
    temp_df = spark.read.option("header","false").option("delimiter",";").csv("donnees/Test/"+data_file,schema=schema)
    df_test = temp_df
df_test = df_test.dropna()
df_test.count()


# In[7]:


from sklearn.model_selection import train_test_split
df = df.toPandas()

X_train, X_test, y_train, y_test = train_test_split(df[['speed','velocity','x','y','z']],df.label,train_size=0.9)


# In[10]:


df_test = df_test.toPandas()


# In[13]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[15]:


model.fit(X_train,y_train)


# In[18]:


r = model.predict(X_test)


# In[22]:


import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

r


# In[27]:


y_test


# In[32]:


import pandas as pd
pd.set_option('display.max_rows', None)
print(y_test)


# In[ ]:




