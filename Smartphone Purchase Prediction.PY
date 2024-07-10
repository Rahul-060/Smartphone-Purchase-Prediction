'''You have been tasked with creating a decision tree model to predict whether a person is likely to purchase a new smartphone based on their age, income, and education level. You are provided with a dataset
containing these attributes and the target variable indicating whether the person made a purchase or not.'''
# this can be solved by KNN
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

# step-2 from data
X=np.array([[25, 50000, 3], [35, 90000,2], [40,60000,5],[45,80000,3],[20,30000,2], [55,120000,4],[28,40000,1], [32,100000,3],[38,75000,2]]) # 1:Btech/MBBS ,2:Master's ,3:PHD's and so on
y = np.array([0,1,1,0,1,0,1,0,1]) # 1:yes ,0:No for purchase

# step-3
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# step-4
model=DecisionTreeClassifier()
model.fit(X_train,y_train)

# Accuracy score checking
y_prediction=model.predict(X_test)
accuracy=accuracy_score(y_test,y_prediction)
print(f"Accuracy score of the model is: {accuracy}")

# user-input
age=float(input("Enter age: "))
income=float(input("Enter Income: "))
Education_level=float(input("Enter Education level: "))
user_data=np.array([[age,income,Education_level]])
prediction=model.predict(user_data)

if prediction[0]==1:
    print("The user is likely to purchase a smartphone")
else:
    print("The user is unlikely to purchase a smartphone")
    

