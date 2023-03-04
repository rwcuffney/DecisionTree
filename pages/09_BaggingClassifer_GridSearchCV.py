import streamlit as st
import myFunctions as my

#models
from sklearn.ensemble import BaggingClassifier






model =  BaggingClassifier() 
general_name = 'Bagging Classifier'

#best_params = {'max_features': 1.0, 'max_samples': 1.0, 'n_estimators': 50}
best_params = {'max_features': 1.0, 'max_samples': .5, 'n_estimators': 100}
model.set_params(**best_params)


my.page_header(general_name,model)

st.write('the parameters for the above model were optimized using GridSearchCV. The code for that process can be seen below.  Ultimately the best parameters were: max_features= 1.0, max_samples=1.0, n_estimators=0')


code = '''
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer, precision_score

# define the scoring function
scoring = make_scorer(precision_score, pos_label=1)

# Generate data:
X_train, X_test, y_train, y_test = custom_train_test_split(df=df,Fraud=5_000,Non_Fraud=90_000)

# Define the parameter grid to search over
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_samples': [0.5, 1.0],
    'max_features': [0.5, 1.0],
}

# Create a BaggingClassifier object
clf = BaggingClassifier()

# Create a GridSearchCV object to search over the parameter grid
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Print the best set of hyperparameters and corresponding score
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")


# Set the best set of hyperparameters as the parameters for the BaggingClassifier
best_params = grid_search.best_params_
clf.set_params(**best_params)

# Fit the BaggingClassifier with the best hyperparameters to the data
clf.fit(X_train, y_train)

# Make predictions on the data and compute the accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
'''
st.code(code)

