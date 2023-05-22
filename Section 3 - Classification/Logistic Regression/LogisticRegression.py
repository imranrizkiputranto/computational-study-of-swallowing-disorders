"""
Logistic Regression Model
Imran Rizki Putranto
22 May 2023

Creating a logistic regression model to determine likelihood of a customer to purchase a brand new SUV
"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
from matplotlib.colors import ListedColormap
matplotlib.use('TkAgg')
np.set_printoptions(threshold=sys.maxsize)

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Splitting into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()  # between -3 to 3
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)  # Imagine x_test and y_test are observations of purchasing new SUVs

# Training logistic model on training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)

# Predicting new results
# Y_pred_new = classifier.predict([X_train[0]])
Y_pred_new = classifier.predict(sc.transform([[30, 87000]]))
Y_pred_new_prob = classifier.predict_proba([X_test[0]])
print(Y_pred_new, Y_pred_new_prob)

# Predicting test set results
Y_test_pred = classifier.predict(X_test)
print(np.concatenate((Y_test_pred.reshape(len(Y_test_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))

# Confusion matrix
confusion_matrix = confusion_matrix(Y_test, Y_test_pred)
print(confusion_matrix)
# [correct preds[0], wrong preds[1(bought in reality)], wrong preds[0(didnt buy in reality)], correct preds[1]]

accuracy_score = accuracy_score(Y_test, Y_test_pred)
print(accuracy_score)  # Accuracy of logistic regression model

# Visualising training set results
X_set, y_set = sc.inverse_transform(X_train), Y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
X_set, y_set = sc.inverse_transform(X_test), Y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
