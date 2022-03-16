#Analyis project
#Author : Prasanth Sukumar

import pandas as pd

df = pd.read_csv('iris.data', header=None)
print(df)
print(df.dtypes)
print(df.columns)

df=df.rename(columns={df.columns[0]: "sepal_length_cm", df.columns[1]: "sepal_width_cm",df.columns[2]: "petal_length_cm", df.columns[3]: "petal_width_cm", df.columns[4]: "class"})
print(df.columns)
print(df)

print(df['class'].value_counts())