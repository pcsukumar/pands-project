#Analyis project
#Author : Prasanth Sukumar

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

names=["sepal_length","sepal_width", "petal_length","petal_width","Species"]
df = pd.read_csv('iris.data', header=None, names=names)
df.head()

print(df)
print(df.dtypes)
print(df.columns)


print(df.columns)
print(df)

print(df['class'].value_counts())


data_desc = df.describe()
print(data_desc)

group_mean = df.groupby('class').mean()
print(group_mean)


group_mean_etc = df.groupby('class').agg(['count', 'min', 'max', 'mean'])
print(group_mean_etc)