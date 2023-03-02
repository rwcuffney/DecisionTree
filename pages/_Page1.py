import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
from transformers import pipeline
from PIL import Image
import requests

st.write("This is page one")

dataset1 = pd.read_csv('./data/fraudTest.csv')
dataset2 = pd.read_csv('./data/fraudTrain.csv')
frames = [dataset1,dataset2]
dataset = pd.concat(frames,ignore_index=True)

st.dataframe(dataset.head(5))
