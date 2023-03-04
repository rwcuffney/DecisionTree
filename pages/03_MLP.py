import streamlit as st
import myFunctions as my

#models:
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

model = MLPClassifier(max_iter=1000, alpha=1, random_state=0)

general_name = 'Multi layer perceptron (MLP)'
#model_name = type(model).__name__


my.page_header(general_name,model)


