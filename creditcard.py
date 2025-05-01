# importing the rquired dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# loading the dataset to Pandas DataFrame
credit_card_data = pd.read_csv(r"C:\Users\Rathika Gupta\OneDrive\Desktop\Python\creditcard.csv")

# distribution of legit and fraudulent transaction: 0 for legit and 1 for fraudulent
print(credit_card_data['Class'].value_counts())

# separating the data for analysis; the below mentioned is the pandas series data type
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)

# undersampling
legit_sample = legit.sample(n = 492) # will take random 492 data points in the majority class which is legit class

# concatenating the two equally sized dataframes
new_data = pd.concat([legit_sample, fraud], axis = 0) # axis = 0 for row-wise concatenation and axis = 1 for column wise concatenation

print(new_data['Class'].value_counts())

# splitting the dataset into features and targets; X --> feature, Y --> target
X = new_data.drop(columns = 'Class', axis = 1) #every column except the class col
Y = new_data['Class']

# splitting the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# training the regression model with training data
model = LogisticRegression(solver='liblinear',max_iter=1000)
model.fit(X_train, Y_train)

# model evaluation:
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)

# 1. Precision score
training_data_precision = precision_score(Y_train, X_train_prediction)
testing_data_precision = precision_score(Y_test, X_test_prediction)

print('Precision on Training data:', training_data_precision)
print('Precision on Testing data:', testing_data_precision)

# 2. F1 score
training_data_f1 = f1_score(Y_train, X_train_prediction)
testing_data_f1 = f1_score(Y_test, X_test_prediction)

print('F1 score on Training data:', training_data_f1)
print('F1 score on Testing data:', testing_data_f1)

# 3. Recall score
training_data_recall = recall_score(Y_train, X_train_prediction)
testing_data_recall = recall_score(Y_test, X_test_prediction)

print('Recall score on Training data:', training_data_recall)
print('Recall score on Testing data:', testing_data_recall)

# 4. Accuracy Score
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score on Training data:', training_data_accuracy)
print('Accuracy score on Testing data:', testing_data_accuracy)
