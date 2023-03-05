import streamlit as st
import myFunctions as my

#models:
from sklearn.naive_bayes import GaussianNB


model = GaussianNB(var_smoothing= 1e-05)


general_name = 'Naive Bayes'

my.page_header(general_name,model,fraud_test_size=.8,non_fraud_test_size=.04)