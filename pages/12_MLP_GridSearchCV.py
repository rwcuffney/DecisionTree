import streamlit as st
import myFunctions as my

#models:
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

params = {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (10,), 'solver': 'sgd'}

model = MLPClassifier(max_iter=1000, random_state=0,**params)

general_name = 'Multi layer perceptron (MLP)'
#model_name = type(model).__name__


my.page_header(general_name,model)


