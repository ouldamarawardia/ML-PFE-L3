#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('git clone https://github.com/Atszvakht/donnees.git')


# In[32]:



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


# In[33]:


df_test = None
data_files1 = os.listdir('donnees\Test')
data_files2 = [s for s in data_files1 if "py" not in s]
for data_file in data_files2:
    print(data_file)
    temp_df = spark.read.option("header","false").option("delimiter",";").csv("donnees/Test/"+data_file,schema=schema)
    df_test = temp_df
df_test = df_test.dropna()
df_test.count()


# In[34]:


df = df.drop('laltitude','longitude','speedAccuracy','directionAccuracy','date_de_recolte')
df_test = df_test.drop('laltitude','longitude','speedAccuracy','directionAccuracy','date_de_recolte','index','label')


# In[35]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy
import pandas as pd


# In[36]:


numpy.random.seed(2)


# In[37]:


X = df.select(["speed","direction","velocity","x","y","z"])
Y = df.select(["label"])
X = X.toPandas()
Y = Y.toPandas()


# In[70]:


model = Sequential()
model.add(Dense(10, input_dim=6, activation='relu')) # input layer requires input_dim param
model.add(Dense(10, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(1, activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1


# In[71]:


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])


# In[72]:


model.fit(X, Y, epochs=5, batch_size=10)


# In[73]:


model.save('weights.h5')


# In[75]:


essaie = model.predict(df_test)


# In[76]:


for index, row in df_test.iterrows():
    print("X=%s, Predicted=%s" % (df_test.iloc[index], essaie[index]))


# In[ ]:




