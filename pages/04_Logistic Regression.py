import streamlit as st
import myFunctions as my

#models:
from sklearn.linear_model import LogisticRegression


model =  logreg = LogisticRegression(C= 0.1, penalty= 'l2')
general_name = 'Logistic Regression'

my.page_header(general_name,model,fraud_test_size=.42,non_fraud_test_size=.96)

st.write('the parameters for the above model were optimized using GridSearchCV. The code for that process can be seen below.  Ultimately the best parameters were: C= 0.1, penalty= l2')


code= '''
# define the scoring function
scoring = make_scorer(precision_score, pos_label=1)

# Generate data:
X_train, X_test, y_train, y_test = custom_train_test_split(df=dataset)

# Define the hyperparameter grid to search over
param_grid = {'C': [0.1, 1.0, 10.0],
              'penalty': ['l1', 'l2']}

# Define the model
clf = LogisticRegression()


# define the scoring function
scoring = make_scorer(precision_score, pos_label=1)

# Define the GridSearchCV object with precision score as the metric
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=scoring)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the corresponding score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best hyperparameters:", best_params)
print("Best precision score:", best_score)

# Fit the best estimator on the entire training set
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)

# Evaluate the performance of the best estimator on the test set
score = best_clf.score(X_test, y_test)
print("Test set score:", score)
'''
st.code(code)