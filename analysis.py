#Analyis project
#Author : Prasanth Sukumar

import pandas as pd

df = pd.read_csv('iris.data', header=None, names=["sepal_length_cm","sepal_width_cm", "petal_length_cm","petal_width_cm","class"])

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