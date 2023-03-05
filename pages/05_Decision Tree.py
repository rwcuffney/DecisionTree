import streamlit as st
import myFunctions as my

#models:
from sklearn.tree import DecisionTreeClassifier


#model =  DecisionTreeClassifier(splitter= 'best', min_samples_split= 20, min_samples_leaf=40, max_depth=2, criterion= 'gini',random_state=0, class_weight='balanced') 
model = DecisionTreeClassifier(max_depth=8,min_samples_leaf=2,min_samples_split=5)

general_name = 'Decision Tree'

my.page_header(general_name,model,fraud_test_size=.55,non_fraud_test_size=.2)