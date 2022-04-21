#Analyis project
#Author : Prasanth Sukumar

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

names=["sepal_length","sepal_width", "petal_length","petal_width","Species"]
df = pd.read_csv('iris.data', header=None, names=names)

dfhead = df.head() #to see the first 5 rows of the data frame

with open("results.txt", "wt") as f: 
     f.write("Head of data frame" + "\n" + str(dfhead) + "\n"*4) #"\n"*3 is to add 3 blank lines

dfshape = df.shape #to view the shape of the data frame, howmany rows and columns.

#dfinfo = df.info() #Display number of rows, columns

with open("results.txt", "a") as f: 
     f.write("Shape of data frame" + "\n" + str(dfshape) + "\n"*4) 

data_desc = df.describe()

with open("results.txt", "a") as f: 
    f.write("Decripton of data frame" + "\n" + str(data_desc) + "\n"*4) 


group_mean = df.groupby('Species').mean()

with open("results.txt", "a") as f: 
    f.write("Group mean for each species" + "\n" + str(group_mean) + "\n"*4) 


group_mean_etc = df.groupby('Species').agg(['count', 'min', 'max', 'mean'])
with open("results.txt", "a") as f: 
    f.write("Group count/min/max/mean for each species" + "\n" + str(group_mean_etc) + "\n"*4) 



#References
# Printing multiple blank lines in python https://stackoverflow.com/questions/28130508/printing-multiple-blank-lines-in-python