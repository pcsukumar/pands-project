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
     f.write("Shape of data frame" + "\n" + str(dfshape) + "\n"*4) #Append the table to result.txt file

data_desc = df.describe() # #to get the descripton of the dataframe

with open("results.txt", "a") as f: 
    f.write("Decripton of data frame" + "\n" + str(data_desc) + "\n"*4) 


group_mean = df.groupby('Species').mean() #This is to generate a table for mean values for each species.

with open("results.txt", "a") as f: 
    f.write("Group mean for each species" + "\n" + str(group_mean) + "\n"*4) 


group_mean_etc = df.groupby('Species').agg(['count', 'min', 'max', 'mean'])#Generate count, min, max, and mean for each species.
with open("results.txt", "a") as f: 
    f.write("Group count/min/max/mean for each species" + "\n" + str(group_mean_etc) + "\n"*4) 


#Plotting data
#Barchart representing the distribution sepal length

x = df.sepal_length
plt.hist(x, bins = 20, color = "green")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Count")
plt.savefig('Barchart_Sepal_Length.png')
plt.show()
   

#Barchart representing the distribution sepal width

x = df.sepal_width
plt.hist(x, bins = 20, color = "Blue")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Count")
plt.savefig('Barchart_Sepal_Width.png')
plt.show()


#Barchart representing the distribution petal length
x = df.petal_length
plt.hist(x, bins = 20, color = "cyan")
plt.title("Petal Length in cm")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Count")
plt.savefig('Barchart_Petal_Length.png')
plt.show()

#Barchart representing the distribution petal width
x = df.petal_width
plt.hist(x, bins = 20, color = "red")
plt.title("Petal Width in cm")
plt.xlabel("Petal Width (cm)")
plt.ylabel("Count")
plt.savefig('Barchart_Petal_Width.png')
plt.show()

#Pair plot to explore the pair wise relationship between different variables 
sns.pairplot(df, hue = 'Species', diag_kind="hist", corner=True) 
plt.savefig('Pairplot_all_variables.png')
plt.show()

#Table exploring the correlation between variables
print('Correlation:')
df.corr()


#Create a correlaton heat map for all species
sns.heatmap(df.corr(), cmap = 'coolwarm', annot=True)
plt.savefig('Correlation_heatmap_all_species.png')
plt.show()



#Species wise analysis

#Correlation between the variables for the species Iris-setosa

setosa = df[df.Species == 'Iris-setosa']
setosa.head()

#Correlation heat map for the species Iris-setosa
sns.heatmap(setosa.corr(), cmap = 'coolwarm', annot=True)
plt.savefig('Correlation_heatmap_setosa.png')
plt.show()


#Correlation between the variables for the species Iris-versicolor
versicolor = df[df.Species == 'Iris-versicolor']
versicolor.head()

#correlation heatmap for the species the species Iris-versicolor
sns.heatmap(versicolor.corr(), cmap = 'coolwarm', annot=True)
plt.savefig('Correlation_heatmap_versicolor.png')
plt.show()

#Correlation between the variables for the species Iris-virginica
virginica = df[df.Species == 'Iris-virginica']
virginica.head()

#Correlation heatmap for the species Iris-virginica
sns.heatmap(virginica.corr(), cmap = 'coolwarm', annot=True)
plt.savefig('Correlation_heatmap_virginica.png')
plt.show()

#References
# Printing multiple blank lines in python https://stackoverflow.com/questions/28130508/printing-multiple-blank-lines-in-python