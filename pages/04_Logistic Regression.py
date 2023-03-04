import streamlit as st
import myFunctions as my

#models:
from sklearn.linear_model import LogisticRegression


model =  logreg = LogisticRegression(max_iter=200)
general_name = 'Logistic Regression'

my.page_header(general_name,model)