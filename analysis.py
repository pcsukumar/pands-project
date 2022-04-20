#Analyis project
#Author : Prasanth Sukumar

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

names=["sepal_length","sepal_width", "petal_length","petal_width","Species"]
df = pd.read_csv('iris.data', header=None, names=names)

dfhead = df.head() #to see the first 5 rows of the data frame

with open("results.txt", "wt") as f: 
     f.write("Head of data frame" + "\n" + str(dfhead) + "\n") 

dfshape = df.shape #to view the shape of the data frame, howmany rows and columns.

#dfinfo = df.info() #Display number of rows, columns

with open("results.txt", "a") as f: 
     f.write("\n" + "Shape of data frame" + "\n" + str(dfshape) + "\n") 

data_desc = df.describe()

with open("results.txt", "a") as f: 
    f.write("\n" + "Decripton of data frame" + "\n" + str(data_desc) + "\n") 


group_mean = df.groupby('Species').mean()

with open("results.txt", "a") as f: 
    f.write("\n" + "Group mean for each species" + "\n" + str(group_mean) + "\n") 


group_mean_etc = df.groupby('Species').agg(['count', 'min', 'max', 'mean'])
with open("results.txt", "a") as f: 
    f.write("\n" + "Group count/min/max/mean for each species" + "\n" + str(group_mean_etc) + "\n") 
