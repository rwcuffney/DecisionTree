import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
import requests
import myFunctions as my
import seaborn as sns

st.header("Understanding the Data")

dataset = my.myData()

n= st.slider('Select a value',value=10,max_value=8000)
st.dataframe(dataset.sample(n))

st.header('Dataset Balance')
'''
shows the distribution of the 'is_fraud' feature
'''


### COLUMNS ####

col1, col2= st.columns([1,3],gap="small")

with col1:
   st.write(dataset['is_fraud'].value_counts())

with col2:
    ####
    df = dataset
    values = df['is_fraud'].value_counts().keys().tolist()
    counts = df['is_fraud'].value_counts().tolist()   
    
    # declaring data
    data = counts
    keys = ['Not Fraud','Fraud']
  
    fig, ax = plt.subplots(figsize=(5,5))
    # declaring exploding pie
    explode = [0, 0.5]
    # define Seaborn color palette to use
    palette_color = sns.color_palette('dark')
  
    # plotting data on chart
    ax.pie(data,labels=keys, colors=palette_color,
        explode=explode, autopct='%.0f%%')
  
    # displaying chart
    st.pyplot(fig)

### END COLUMNS. ####


st.header('Dtypes')

'''
We need to have the values stored in an appropriate data type. Otherwise, we may encounter errors. For large datasets, memory usage is greatly affected by correct data type selection. For example, the “categorical” data type is more appropriate than the “object” data type for categorical data, especially when the number of categories is much less than the number of rows.

Dtypes shows the data type of each column.
'''

st.write(dataset.dtypes)


st.header('Shape and Size')
'''
The shape can be used on numpy arrays, pandas series, and data frames. It shows the number of dimensions and the size of each dimension.

Since data frames are two-dimensional, what shape returns is the number of rows and columns. It measures how much data we have and is a key input to the data analysis process.

Furthermore, the ratio of rows and columns is very important when designing and implementing a machine-learning model. If we do not have enough observations (rows) concerning features (columns), we may need to apply some pre-processing techniques such as dimensional reduction or feature extraction.
'''
st.write(f'Dataset shape: {dataset.shape}')
st.write(f'Dataset size: {dataset.size}')


st.header('describe( )')
'''
If there’s one thing, you do repeatedly in the process of exploratory data analysis — performing a statistical summary for every (or almost every) attribute.

It would be a tedious process without the right tools — but thankfully, Pandas is here to do the heavy lifting for you. The describe() method will do a quick statistical summary for every numerical column, as shown below:
'''
st.dataframe(dataset.describe())
'''
Now I’m using the transpose operator to switch from columns to rows, and vice-versa.
'''

dataset.describe(include='all').T

st.header('Identifying Missing Values Isnull')
'''
Handling missing values is a critical step in building a robust data analysis process. The missing values should be a top priority since they have a significant effect on the accuracy of any analysis.
'''

st.write(dataset.isnull())





