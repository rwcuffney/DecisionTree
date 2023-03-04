import streamlit as st
import myFunctions as my

#models:
from sklearn.ensemble import BaggingClassifier

model =  BaggingClassifier() 


general_name = 'Bagging Classifier'

my.page_header(general_name,model)