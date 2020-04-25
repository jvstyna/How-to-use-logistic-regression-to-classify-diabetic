# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 12:31:10 2020

@author: Królowa J
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
import seaborn as sns

csv_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
diab_data = pd.read_csv(csv_path)

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

X = diab_data[feature_names]
y = diab_data.Outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

lregr=LogisticRegression()
lregr.fit(X_train, y_train)
pred_1 = lregr.predict(X_test)


conf_matrix = metrics.confusion_matrix(y_test, pred_1)
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Original Diabetes True Values    : {0} ({1:0.2f}%)".format(len(diab_data.loc[diab_data['Outcome'] == 1]), (len(diab_data.loc[diab_data['Outcome'] == 1])/len(diab_data.index)) * 100))
print("Original Diabetes False Values   : {0} ({1:0.2f}%)".format(len(diab_data.loc[diab_data['Outcome'] == 0]), (len(diab_data.loc[diab_data['Outcome'] == 0])/len(diab_data.index)) * 100))
print("")
print("Training Diabetes True Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))
print("Training Diabetes False Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))
print("")
print("Test Diabetes True Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))
print("Test Diabetes False Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))
print("")
 
db = diab_data.corr()

def plot_corr(df, size=11):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    
plot_corr(db)