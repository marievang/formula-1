#####################################################################3
import pandas as pd
import os

path_to_files = "/home/Desktop/linux/Python/chav/final_project/Downloaded_data"
os.chdir(path_to_files)

#Import data
dataset_curr_dr = pd.read_csv("current_driver_dataset.csv")
dataset_curr_dr = dataset_curr_dr.drop(["code", "constructorRef"], axis=1)

#Find podium
dataset_curr_dr.loc[:, "y"] = dataset_curr_dr["race_position"].apply(lambda x: 1 if x in [1, 2, 3] else 0)

#Find winner
#dataset_curr_dr.loc[:, "y"] = dataset_curr_dr["race_position"].apply(lambda x: 1 if x == 1 else 0)

#################################################
# Split the data into features (X) and target (y)
X = dataset_curr_dr.drop(["y",'race_position',"number"], axis=1)
y = dataset_curr_dr[ "y"]


#PODIUM
#give new dataset for prediction
latest_races = pd.read_csv("fixed_dataset.csv")
correct_results = pd.read_csv("podium.csv")
correct_results = correct_results.drop(["code"],axis=1)
#Merge the race data with the results
latest_races = latest_races.merge(correct_results, on=["raceId", "driverId"])

#concatenate the lastest race results to the training dataset
X = pd.concat([X, (latest_races.drop(["race_position"], axis=1))])
y = pd.concat([y, latest_races["race_position"]])

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

#Test model
rf = RandomForestClassifier( random_state=2)
rf.fit(X_train, y_train, )
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Hyperparameter tuning
############ HYPERPARAMETER  TUNING	####################

param_dist = {
    "max_features": [1, 2, 3, 5, None],
    "max_leaf_nodes": [10, 100, 1000, None],
    "min_samples_leaf": [1, 2, 5, 10, 20, 50, 100],
}

# Create a random forest classifier
rf = RandomForestClassifier(random_state=2)

# Use random search to find the best hyperparameters
search_cv = RandomizedSearchCV(rf,
    param_distributions=param_dist,
    scoring="neg_mean_absolute_error",
    n_iter=10,
    random_state=2,
    n_jobs=2,
)
search_cv.fit(X_train, y_train)
# Create a variable for the best model
best_rf = search_cv.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  search_cv.best_params_)

# Generate predictions with the best model
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_pred, y_test)) 

################################################################

#################
#FEATURE IMPORTANCE
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

feature_names = list(X.columns)
forest = RandomForestClassifier(min_samples_leaf= 5, max_leaf_nodes= 10, max_features= None, random_state=2)
forest.fit(X_train, y_train)

from sklearn.inspection import permutation_importance

start_time = time.time()
result = permutation_importance(forest, X_test, y_test, n_repeats=100, random_state=2, n_jobs=2)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=feature_names)

#Features with negative importance
print("Features with negative importance:")
print(forest_importances[forest_importances < 0].sort_values(ascending=False))

print("Features with positive importance:")
print(forest_importances[forest_importances > 0].sort_values(ascending=False))

print("Features with zero importance:")
print(forest_importances[forest_importances == 0].sort_values(ascending=False))

print(f"Totals: negative {sum(forest_importances < 0)}, positive {sum(forest_importances > 0)}, zero {sum(forest_importances == 0)}")


fig, ax = plt.subplots()
bars = forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()

# Add values on each bar
for bar in bars.patches:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', 
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.show()

#remove features with negative importance
X = X.drop(["constructor_points", "constructor_position", "constructor_wins", "circuitId"], axis=1)

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

#NEW Hyperparameter tuning
import numpy as np
from sklearn.ensemble import RandomForestRegressor
############ HYPERPARAMETER  TUNING	####################
#param_dist = {'n_estimators': randint(50,500), 'max_depth': randint(1,20), 'bootstrap': [True, False]}
param_dist = {
    "max_features": [1, 2, 3, 5, None],
    "max_leaf_nodes": [10, 100, 1000, None],
    "min_samples_leaf": [1, 2, 5, 10, 20, 50, 100],
}

# Create a random forest classifier
rf = RandomForestClassifier(random_state=2)

# Use random search to find the best hyperparameters
search_cv = RandomizedSearchCV(rf,
    param_distributions=param_dist,
    scoring="neg_mean_absolute_error",
    n_iter=10,
    random_state=2,
    n_jobs=2,
)
search_cv.fit(X_train, y_train)
# Create a variable for the best model
best_rf = search_cv.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  search_cv.best_params_)

# Generate predictions with the best model
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_pred, y_test)) 


#################################
#cross validation error for random forest classifier
from sklearn.model_selection import cross_val_score

# Define rf_model and X_processed
rf_model = RandomForestClassifier(min_samples_leaf= 5, max_leaf_nodes= 100, max_features= 3, random_state=2)
rf_model.fit(X_train, y_train)
X_processed = X_train  # Assuming X_processed is the same as X_train for this example

rf_scores = cross_val_score(rf_model, X_processed, y_train, cv=5, scoring='accuracy')

print("Random Forest Cross Validation Scores:")
print(rf_scores)

#cross validation error for random forest classifier with best hyperparameters

###################################################################################################################
#WINNER prediction
#####################################################################3
import pandas as pd
import os

path_to_files = "/home/emikot/Desktop/linux/Python/chav/final_project/Downloaded_data"
os.chdir(path_to_files)

#Import data
dataset_curr_dr = pd.read_csv("current_driver_dataset.csv")
dataset_curr_dr = dataset_curr_dr.drop(["code", "constructorRef"], axis=1)


#Find winner
dataset_curr_dr.loc[:, "y"] = dataset_curr_dr["race_position"].apply(lambda x: 1 if x == 1 else 0)

#################################################
# Split the data into features (X) and target (y)
X = dataset_curr_dr.drop(["y",'race_position',"number"], axis=1)
y = dataset_curr_dr[ "y"]



#WINNER
#give new dataset for prediction
latest_races = pd.read_csv("fixed_dataset.csv")
correct_results = pd.read_csv("winner.csv")
correct_results = correct_results.drop(["code"],axis=1)
#Merge the race data with the results
latest_races = latest_races.merge(correct_results, on=["raceId", "driverId"])

#concatenate the lastest race results to the training dataset
X = pd.concat([X, (latest_races.drop(["race_position"], axis=1))])
y = pd.concat([y, latest_races["race_position"]])

#Split the data into training and test sets

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

#Test model
rf = RandomForestClassifier( random_state=2)
rf.fit(X_train, y_train, )
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Hyperparameter tuning
import numpy as np
from sklearn.ensemble import RandomForestRegressor
############ HYPERPARAMETER  TUNING	####################
#param_dist = {'n_estimators': randint(50,500), 'max_depth': randint(1,20), 'bootstrap': [True, False]}
param_dist = {
    "max_features": [1, 2, 3, 5, None],
    "max_leaf_nodes": [10, 100, 1000, None],
    "min_samples_leaf": [1, 2, 5, 10, 20, 50, 100],
}

# Create a random forest classifier
rf = RandomForestClassifier(random_state=2)

# Use random search to find the best hyperparameters
search_cv = RandomizedSearchCV(rf,
    param_distributions=param_dist,
    scoring="neg_mean_absolute_error",
    n_iter=10,
    random_state=2,
    n_jobs=2,
)
search_cv.fit(X_train, y_train)
# Create a variable for the best model
best_rf = search_cv.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  search_cv.best_params_)

# Generate predictions with the best model
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_pred, y_test)) 

################################################################

#################################
#FEATURE IMPORTANCE
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

feature_names = list(X.columns)
forest = RandomForestClassifier(min_samples_leaf= 5, max_leaf_nodes= 10, max_features= None, random_state=2)
forest.fit(X_train, y_train)

from sklearn.inspection import permutation_importance

start_time = time.time()
result = permutation_importance(forest, X_test, y_test, n_repeats=100, random_state=2, n_jobs=2)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=feature_names)

#Features with negative importance
print("Features with negative importance:")
print(forest_importances[forest_importances < 0].sort_values(ascending=False))

print("Features with positive importance:")
print(forest_importances[forest_importances > 0].sort_values(ascending=False))

print("Features with zero importance:")
print(forest_importances[forest_importances == 0].sort_values(ascending=False))

print(f"Totals: negative {sum(forest_importances < 0)}, positive {sum(forest_importances > 0)}, zero {sum(forest_importances == 0)}")


fig, ax = plt.subplots()
bars = forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()

# Add values on each bar
#for bar in bars.patches:
#    height = bar.get_height()
#    ax.annotate(f'{height:.3f}', 
#                xy=(bar.get_x() + bar.get_width() / 2, height),
#                xytext=(0, 3),  # 3 points vertical offset
#                textcoords="offset points",
#                ha='center', va='bottom')
#
plt.show()

#remove features with negative importance
X = X.drop(["constructor_wins", "circuitId", "standings_wins", "DriverErrorRate"], axis=1)

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

#NEW Hyperparameter tuning
import numpy as np
from sklearn.ensemble import RandomForestRegressor
############ HYPERPARAMETER  TUNING	####################
#param_dist = {'n_estimators': randint(50,500), 'max_depth': randint(1,20), 'bootstrap': [True, False]}
param_dist = {
    "max_features": [1, 2, 3, 5, None],
    "max_leaf_nodes": [10, 100, 1000, None],
    "min_samples_leaf": [1, 2, 5, 10, 20, 50, 100],
}

# Create a random forest classifier
rf = RandomForestClassifier(random_state=2)

# Use random search to find the best hyperparameters
search_cv = RandomizedSearchCV(rf,
    param_distributions=param_dist,
    scoring="neg_mean_absolute_error",
    n_iter=10,
    random_state=2,
    n_jobs=2,
)
search_cv.fit(X_train, y_train)
# Create a variable for the best model
best_rf = search_cv.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  search_cv.best_params_)

# Generate predictions with the best model
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_pred, y_test)) 


#################################
#cross validation error for random forest classifier
from sklearn.model_selection import cross_val_score

# Define rf_model and X_processed
rf_model = RandomForestClassifier(min_samples_leaf= 5, max_leaf_nodes= 1000, max_features= 2, random_state=2)
rf_model.fit(X_train, y_train)
X_processed = X_train  # Assuming X_processed is the same as X_train for this example

rf_scores = cross_val_score(rf_model, X_processed, y_train, cv=5, scoring='accuracy')

print("Random Forest Cross Validation Scores:")
print(rf_scores)




import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
# 2. Logistic Regression

#PODIUM
#import data
path_to_files = "/home/emikot/Desktop/linux/Python/chav/final_project/Downloaded_data"
os.chdir(path_to_files)

#Import data
dataset_curr_dr = pd.read_csv("current_driver_dataset.csv")
dataset_curr_dr = dataset_curr_dr.drop(["code", "constructorRef"], axis=1)

#Find podium
dataset_curr_dr.loc[:, "y"] = dataset_curr_dr["race_position"].apply(lambda x: 1 if x in [1, 2, 3] else 0)

#Find winner
#dataset_curr_dr.loc[:, "y"] = dataset_curr_dr["race_position"].apply(lambda x: 1 if x == 1 else 0)

#################################################
# Split the data into features (X) and target (y)
X = dataset_curr_dr.drop(["y",'race_position',"number"], axis=1)
y = dataset_curr_dr[ "y"]

#PODIUM
#give new dataset for prediction
latest_races = pd.read_csv("fixed_dataset.csv")
correct_results = pd.read_csv("podium.csv")
correct_results = correct_results.drop(["code"],axis=1)
#Merge the race data with the results
latest_races = latest_races.merge(correct_results, on=["raceId", "driverId"])

#concatenate the lastest race results to the training dataset
X = pd.concat([X, (latest_races.drop(["race_position"], axis=1))])
y = pd.concat([y, latest_races["race_position"]])

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

#test default model
lr_model = LogisticRegression(random_state=2)
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)

print("Logistic Regression Results:")
print(classification_report(y_test, lr_y_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_y_pred))

################################################################
#Hyperparameter tuning
from sklearn.linear_model import LogisticRegressionCV

model = LogisticRegressionCV(random_state=2).fit(X_train, y_train)
predict = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predict))


#Calculate feature importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE


model = LogisticRegressionCV(random_state=2).fit(X_train, y_train)
# Coefficients and Odds Ratios
coefficients = model.coef_[0]
odds_ratios = np.exp(coefficients)


# Display feature importance using coefficients and odds ratios
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients,
    'Odds Ratio': odds_ratios
})
print("\nFeature Importance (Coefficient and Odds Ratio):")
print(feature_importance.sort_values(by='Coefficient', ascending=False))

# Permutation Importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=2, n_jobs=-1)
perm_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance Mean': perm_importance.importances_mean,
    'Importance Std': perm_importance.importances_std
})
print("\nPermutation Importance:")
print(perm_importance_df.sort_values(by='Importance Mean', ascending=False))

#plot the feature importance from the perm_importance dataframe
import matplotlib.pyplot as plt
perm_importance_df = perm_importance_df.sort_values(by='Importance Mean', ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(perm_importance_df['Feature'], perm_importance_df['Importance Mean'], xerr=perm_importance_df['Importance Std'])
plt.xlabel('Permutation Importance')
plt.ylabel('Feature')
plt.title('Permutation Importance for Podium Prediction')
plt.show()


#Feature selection
X= X.drop("q3", axis=1)

#New hyoerparameter tuning
#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
model = LogisticRegressionCV(random_state=2).fit(X_train, y_train)
predict = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predict))



#cross validation for logistic regression classifier
lr_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print("Logistic Regression Cross Validation Scores:")
print(lr_scores)


#Winner prediction
#import data
path_to_files = "/home/emikot/Desktop/linux/Python/chav/final_project/Downloaded_data"
os.chdir(path_to_files)

#Import data
dataset_curr_dr = pd.read_csv("current_driver_dataset.csv")
dataset_curr_dr = dataset_curr_dr.drop(["code", "constructorRef"], axis=1)

#Find winner
dataset_curr_dr.loc[:, "y"] = dataset_curr_dr["race_position"].apply(lambda x: 1 if x == 1 else 0)

#################################################
# Split the data into features (X) and target (y)
X = dataset_curr_dr.drop(["y",'race_position',"number"], axis=1)
y = dataset_curr_dr[ "y"]

#PODIUM
#give new dataset for prediction
latest_races = pd.read_csv("fixed_dataset.csv")
correct_results = pd.read_csv("winner.csv")
correct_results = correct_results.drop(["code"],axis=1)
#Merge the race data with the results
latest_races = latest_races.merge(correct_results, on=["raceId", "driverId"])

#concatenate the lastest race results to the training dataset
X = pd.concat([X, (latest_races.drop(["race_position"], axis=1))])
y = pd.concat([y, latest_races["race_position"]])

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

#test default model
lr_model = LogisticRegression(random_state=2)
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)

print("Logistic Regression Results:")
print(classification_report(y_test, lr_y_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_y_pred))

################################################################
#Hyperparameter tuning
from sklearn.linear_model import LogisticRegressionCV

model = LogisticRegressionCV(random_state=2).fit(X_train, y_train)
predict = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predict))

#Calculate feature importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE


model = LogisticRegressionCV(random_state=2).fit(X_train, y_train)
# Coefficients and Odds Ratios
coefficients = model.coef_[0]
odds_ratios = np.exp(coefficients)


# Display feature importance using coefficients and odds ratios
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients,
    'Odds Ratio': odds_ratios
})
print("\nFeature Importance (Coefficient and Odds Ratio):")
print(feature_importance.sort_values(by='Coefficient', ascending=False))

# Permutation Importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=2, n_jobs=-1)
perm_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance Mean': perm_importance.importances_mean,
    'Importance Std': perm_importance.importances_std
})
print("\nPermutation Importance:")
print(perm_importance_df.sort_values(by='Importance Mean', ascending=False))


#plot the feature importance from the perm_importance dataframe
import matplotlib.pyplot as plt
perm_importance_df = perm_importance_df.sort_values(by='Importance Mean', ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(perm_importance_df['Feature'], perm_importance_df['Importance Mean'], xerr=perm_importance_df['Importance Std'])
plt.xlabel('Permutation Importance')
plt.ylabel('Feature')
plt.title('Permutation Importance for Winner Prediction')
plt.show()


#Feature selection
X= X.drop("q3", axis=1)

#New hyoerparameter tuning
#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
model = LogisticRegressionCV(random_state=2).fit(X_train, y_train)
predict = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predict))



#cross validation for logistic regression classifier
lr_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print("Logistic Regression Cross Validation Scores:")
print(lr_scores)















#cross validation error for random forest classifier
from sklearn.model_selection import cross_val_score

# Define rf_model and X_processed
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
X_processed = X_train  # Assuming X_processed is the same as X_train for this example

rf_scores = cross_val_score(rf_model, X_processed, y_train, cv=5, scoring='accuracy')
print("Random Forest Cross Validation Scores:")
print(rf_scores)

#cross validation error for random forest classifier with best hyperparameters
rf_model = RandomForestClassifier(max_depth= 2, n_estimators= 219, bootstrap= True)
rf_model.fit(X_train, y_train)
X_processed = X_train  # Assuming X_processed is the same as X_train for this example

rf_scores = cross_val_score(rf_model, X_processed, y_train, cv=5, scoring='accuracy')
print("Random Forest Cross Validation Scores:")
print(rf_scores)


