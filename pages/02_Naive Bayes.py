import streamlit as st
import myFunctions as my

#models:
from sklearn.naive_bayes import GaussianNB


model = GaussianNB()
general_name = 'Naive Bayes'

my.page_header(general_name,model)