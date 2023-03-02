import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
from transformers import pipeline
from PIL import Image
import requests


st.write("This is the home page")

#dataset1 = pd.read_csv('./data/fraudTest.csv')
#dataset2 = pd.read_csv('./data/fraudTrain.csv')
#frames = [dataset1,dataset2]
#dataset = pd.concat(frames,ignore_index=True)

#st.dataframe(dataset.head(5))

# Set the path to the CSV file
csv_path = "/Users/rwcuffney/Documents/NorthWestern_University/Machine_Learning/__Final_Project/_app/data/fraudTrain.csv"
#Set the path to the CSV file

import os

#to get the current working directory
directory = os.getcwd()
st.write(directory)


# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_path)

# Do something with the DataFrame, such as displaying it in your Streamlit app
st.dataframe(df.head(5))