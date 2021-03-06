## Analysis of Fisher’s Iris data set
#### Author: Prasanth Sukumar

### Introduction
The Iris dataset is the data set created by the British biologist Ronald Fisher. This data set contains measurement of sepal length, sepal width, petal length and petal width for 150 flowers from 3 species of Iris flower; viz Iris setosa, Iris versicolor, and Iris virginica. This analysis aims to explore the variables in the dataset and see if there exists any relationship between them.


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
names=["sepal_length","sepal_width", "petal_length","petal_width","Species"] #As the dataset has no column names, a 'names' list is created with column names and add the names when data export from csv.
df = pd.read_csv('iris.data', header=None, names=names)
df.head() #to see the first 5 rows of the data frame
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



## Explore Data

#### Use the 'shape' attribute to check the number of rows and cloumns in the data frame.


```python
df.shape #to view the shape of the data frame, howmany rows and columns.
```




    (150, 5)




```python
df.info() #Display number of rows, columns
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   sepal_length  150 non-null    float64
     1   sepal_width   150 non-null    float64
     2   petal_length  150 non-null    float64
     3   petal_width   150 non-null    float64
     4   Species       150 non-null    object 
    dtypes: float64(4), object(1)
    memory usage: 6.0+ KB
    

#### Check the data types of variables in the data frame


```python
df.dtypes # to see the the data type of each column. 
```




    sepal_length    float64
    sepal_width     float64
    petal_length    float64
    petal_width     float64
    Species          object
    dtype: object



#### Check the number of rows for each species in the data set


```python
df['Species'].value_counts() #to see the number of unique values.
```




    Iris-setosa        50
    Iris-versicolor    50
    Iris-virginica     50
    Name: Species, dtype: int64



#### Describe the numerical variables in the data frame


```python
df.describe() #to get the descripton of the dataframe
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.054000</td>
      <td>3.758667</td>
      <td>1.198667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.433594</td>
      <td>1.764420</td>
      <td>0.763161</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>



Sepal length ranges from 4.3 cm to 7.9 cm with a mean of 5.8 cm and standard deviation 0.82. 
Sepal width ranges from 1 cm to 6.9 cm with a mean of 3.1 cm and standard deviation 0.43. 
Petal length ranges from 1.0 cm to 6.9 cm with a mean 3.75 cm and standard deviation 1.76.
Petal width ranges from 0.1 cm to 2.5 cm with a mean 1.19 cm and standard deviation 0.76.

#### Find the mean for each class


```python
df.groupby('Species').mean() #This is to generate a table for mean values for each species.
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Iris-setosa</th>
      <td>5.006</td>
      <td>3.418</td>
      <td>1.464</td>
      <td>0.244</td>
    </tr>
    <tr>
      <th>Iris-versicolor</th>
      <td>5.936</td>
      <td>2.770</td>
      <td>4.260</td>
      <td>1.326</td>
    </tr>
    <tr>
      <th>Iris-virginica</th>
      <td>6.588</td>
      <td>2.974</td>
      <td>5.552</td>
      <td>2.026</td>
    </tr>
  </tbody>
</table>
</div>



The above table shows mean sepal length, sepal width and petal lenght, petal width for each species of Iris. Virginica has longest mean sepal length and petal length, wheareas setosa has shortest mean sepal length and petal length. Versicolor has shortest mean sepal width, whereas setosa has shortest mean petal width.

#### Find Count, Min, Max and Mean for each class


```python
df.groupby('Species').agg(['count', 'min', 'max', 'mean']) #Generate count, min, max, and mean for each species.
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">sepal_length</th>
      <th colspan="4" halign="left">sepal_width</th>
      <th colspan="4" halign="left">petal_length</th>
      <th colspan="4" halign="left">petal_width</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>count</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>count</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>count</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Iris-setosa</th>
      <td>50</td>
      <td>4.3</td>
      <td>5.8</td>
      <td>5.006</td>
      <td>50</td>
      <td>2.3</td>
      <td>4.4</td>
      <td>3.418</td>
      <td>50</td>
      <td>1.0</td>
      <td>1.9</td>
      <td>1.464</td>
      <td>50</td>
      <td>0.1</td>
      <td>0.6</td>
      <td>0.244</td>
    </tr>
    <tr>
      <th>Iris-versicolor</th>
      <td>50</td>
      <td>4.9</td>
      <td>7.0</td>
      <td>5.936</td>
      <td>50</td>
      <td>2.0</td>
      <td>3.4</td>
      <td>2.770</td>
      <td>50</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>4.260</td>
      <td>50</td>
      <td>1.0</td>
      <td>1.8</td>
      <td>1.326</td>
    </tr>
    <tr>
      <th>Iris-virginica</th>
      <td>50</td>
      <td>4.9</td>
      <td>7.9</td>
      <td>6.588</td>
      <td>50</td>
      <td>2.2</td>
      <td>3.8</td>
      <td>2.974</td>
      <td>50</td>
      <td>4.5</td>
      <td>6.9</td>
      <td>5.552</td>
      <td>50</td>
      <td>1.4</td>
      <td>2.5</td>
      <td>2.026</td>
    </tr>
  </tbody>
</table>
</div>



The table above shows the minimum, maximum and mean sepal length, sepal width, petal length, petal width for each species.

## Plotting Data

### Histograms


```python
x = df.sepal_length
plt.hist(x, bins = 20, color = "green")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Count")
plt.show()
```


    
![png](images/output_22_0.png)
    



```python
x = df.sepal_width
plt.hist(x, bins = 20, color = "Blue")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Count")
plt.show()
```


    
![png](images/output_23_0.png)
    



```python
x = df.petal_length
plt.hist(x, bins = 20, color = "cyan")
plt.title("Petal Length in cm")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Count")
plt.show()
```


    
![png](images/output_24_0.png)
    



```python
x = df.petal_width
plt.hist(x, bins = 20, color = "red")
plt.title("Petal Width in cm")
plt.xlabel("Petal Width (cm)")
plt.ylabel("Count")
plt.show()
```


    
![png](images/output_25_0.png)
    



```python
g = sns.pairplot(df, hue = 'Species', diag_kind="hist", corner=True) #Pair plot to explore the pair wise relationship between different variables 
```


    
![png](images/output_26_0.png)
    


## Correlation analysis


```python
print('Correlation:') #Table exploring the correlation between variables
df.corr()
```

    Correlation:
    




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sepal_length</th>
      <td>1.000000</td>
      <td>-0.109369</td>
      <td>0.871754</td>
      <td>0.817954</td>
    </tr>
    <tr>
      <th>sepal_width</th>
      <td>-0.109369</td>
      <td>1.000000</td>
      <td>-0.420516</td>
      <td>-0.356544</td>
    </tr>
    <tr>
      <th>petal_length</th>
      <td>0.871754</td>
      <td>-0.420516</td>
      <td>1.000000</td>
      <td>0.962757</td>
    </tr>
    <tr>
      <th>petal_width</th>
      <td>0.817954</td>
      <td>-0.356544</td>
      <td>0.962757</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(df.corr(), cmap = 'coolwarm', annot=True) #Create a correlaton heat map for all species
```




   




    
![png](images/output_29_1.png)
    


### Species wise analysis


```python
setosa = df[df.Species == 'Iris-setosa'] #Correlation between the variables for the species Iris-setosa
setosa.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(setosa.corr(), cmap = 'coolwarm', annot=True) #Correlation heat map for the species Iris-setosa
```




 




    
![png](images/output_32_1.png)
    



```python
versicolor = df[df.Species == 'Iris-versicolor'] #Correlation between the variables for the species Iris-versicolor
versicolor.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <td>7.0</td>
      <td>3.2</td>
      <td>4.7</td>
      <td>1.4</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>51</th>
      <td>6.4</td>
      <td>3.2</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>52</th>
      <td>6.9</td>
      <td>3.1</td>
      <td>4.9</td>
      <td>1.5</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>53</th>
      <td>5.5</td>
      <td>2.3</td>
      <td>4.0</td>
      <td>1.3</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>54</th>
      <td>6.5</td>
      <td>2.8</td>
      <td>4.6</td>
      <td>1.5</td>
      <td>Iris-versicolor</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(versicolor.corr(), cmap = 'coolwarm', annot=True) #correlation heatmap for the species Iris-versicolor
```








    
![png](images/output_34_1.png)
    



```python
virginica = df[df.Species == 'Iris-virginica'] #Correlation between the variables for the species Iris-virginica
virginica.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>6.3</td>
      <td>3.3</td>
      <td>6.0</td>
      <td>2.5</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>101</th>
      <td>5.8</td>
      <td>2.7</td>
      <td>5.1</td>
      <td>1.9</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>102</th>
      <td>7.1</td>
      <td>3.0</td>
      <td>5.9</td>
      <td>2.1</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>103</th>
      <td>6.3</td>
      <td>2.9</td>
      <td>5.6</td>
      <td>1.8</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>104</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.8</td>
      <td>2.2</td>
      <td>Iris-virginica</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(virginica.corr(), cmap = 'coolwarm', annot=True) #Correlation heatmap for the species Iris-virginica
```





    
![png](images/output_36_1.png)
    



```python
plt.figure(figsize=(12,7))
plt.title('Comparison between petal width and length for each species') #Comparison between petal width and length for each species
sns.scatterplot(x=df.petal_length, y=df.petal_width, hue = df.Species)
```

    
![png](images/output_37_1.png)
    


The above scatter plot shows that there is a good positive correlation between petal width and petal length. This relationship is most prominent in the case of Iris-versicolor. Correlation between petal width and petal length is not that promenent in the case of other species.


```python
plt.figure(figsize=(12,7))
plt.title('Comparison between sepal width and length for each species') #Comparison between petal width and length for each species
sns.scatterplot(x=df.sepal_length, y=df.sepal_width, hue = df.Species)
```




      
![png](images/output_39_1.png)
    


The above scatter plot shows that there is a positive correlation between sepal width and sepal length in the case of Iris-setosa only. Correlation between sepal width and sepal length is not clearly observable in the case of other species.

### Summary and Conclusion

Analysis shows that the average petal length and petal width is comparatively much smaller for the species Iris Setosa (1.46 cm and 0.244 cm respectively) and largest for Iris Virginica (5.55 cm and 2.03 cm). Iris Versicolor has almost similar petal length and width (4.26 cm and 1.33 cm) to that of the Iris Versicolor.

Standard deviation of petal measurements shows that Iris Setosa has much less variability and Iris Virginica has the highest variability. Even though Iris Setosa has the smallest average speal lengths, the difference between the three species is minimal. Standard deviation shows that Setosa has the lowest variability in sepal length and Virginica has the highest.

The average sepal width of the Setosa is slightly more than the averages for the other two species., however standard deviation shows that they all have almost same variability (~0.3). The scatter plot shows a strong positive relationship between the petal length and petal width for all species. Correlation analysis shows the correlation between petal length and petal width 0.96. However, when analyse this relationship for each species, it can be found that the correlation is weakest for Iris Setosa (0.31) and highest for Virginica (0.71). The correlation matrix shows a strong relationship between sepal length and sepal width for the Iris Setosa only.

### References
Iris flower data set. https://en.wikipedia.org/wiki/Iris_flower_data_set

Add column names to dataframe in Pandas: https://www.geeksforgeeks.org/add-column-names-to-dataframe-in-pandas/

pandas: Get the number of rows, columns, all elements (size) of DataFrame: https://note.nkmk.me/en/python-pandas-len-shape-size/

Python | Pandas Dataframe.describe() method https://www.geeksforgeeks.org/python-pandas-dataframe-describe-method/

Use Pandas Groupby to Group and Summarise DataFrames https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

GroupBy in Pandas: Your Guide to Summarizing and Aggregating Data in Python https://www.analyticsvidhya.com/blog/2020/03/groupby-pandas-aggregating-data-python/

Plotting a Histogram in Python with Matplotlib and Pandas https://datagy.io/histogram-python/

Seaborn pairplot example https://pythonbasics.org/seaborn-pairplot/

How to Create a Seaborn Correlation Heatmap in Python? https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e

How To Filter Pandas Dataframe By Values of Column? https://cmdlinetips.com/2018/02/how-to-subset-pandas-dataframe-based-on-values-of-a-column/

Printing multiple blank lines in python https://stackoverflow.com/questions/28130508/printing-multiple-blank-lines-in-python