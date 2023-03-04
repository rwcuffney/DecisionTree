import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
from transformers import pipeline
from PIL import Image
import requests

st.set_page_config(layout="wide")

st.title("Credit Card Fraud Detection with Machine Learning")


