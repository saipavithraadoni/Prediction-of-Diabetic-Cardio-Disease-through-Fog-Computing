# -*- coding: utf-8 -*-
"""
Created on Tue May  3 12:48:31 2022

@author: adoni
"""
from os import sep
from pickle import TRUE
import pandas as pd  
import csv                                                     
from sklearn import preprocessing                                        
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB   
from sklearn.tree import DecisionTreeClassifier   
from sklearn.svm import SVC
import numpy as np                          
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from csv import writer





from flask import Flask,render_template,request
app = Flask(__name__,template_folder='templates')
@app.route('/',methods=['POST','GET'])
def calculate():
   if (request.method=='POST' and 'page' in request.form):
       """CARDIO........"""
       age=int(request.form.get('page'))
       name=str(request.form.get('pname'))
       sex=int(request.form.get('gender'))
       cp=int(request.form.get('cpt'))
       trestbps=int(request.form.get('rbp'))
       chol=int(request.form.get('chol'))
       fbs=float(request.form.get('fbs'))
       restecg=int(request.form.get('rcg'))
       thalach=int(request.form.get('thalach'))
       exang=int(request.form.get('eia'))
       oldpeak=int(request.form.get('op'))
       slope=float(request.form.get('slope'))
       ca=int(request.form.get('nov'))
       thal=int(request.form.get('thal'))


       heart_data = pd.read_csv('heart.csv')
       heart_data['target'].value_counts()
       X = heart_data.drop(columns='target', axis=1)
       Y = heart_data['target']


       X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, stratify=Y, random_state=2)
       print(X.shape, X_train.shape, X_test.shape)
       

       model = LogisticRegression()
       model.fit(X_train, Y_train)
       X_train_prediction = model.predict(X_train)
       lr_training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
       lr_training_data_accuracy="{:.2f}".format(lr_training_data_accuracy*100)
       print('Accuracy on Training data : ', lr_training_data_accuracy)
       X_test_prediction = model.predict(X_test)
       lr_test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
       lr_test_data_accuracy="{:.2f}".format(lr_test_data_accuracy*100)
       print('Accuracy on Test data : ', lr_test_data_accuracy)




       model = KNeighborsClassifier(n_neighbors=5)                                                                    
       model.fit(X_train, Y_train)
       X_train_prediction = model.predict(X_train)
       knn_training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
       knn_training_data_accuracy="{:.2f}".format(knn_training_data_accuracy*100)
       print('Accuracy on Training data : ', knn_training_data_accuracy)
       X_test_prediction = model.predict(X_test)
       knn_test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
       knn_test_data_accuracy="{:.2f}".format(knn_test_data_accuracy*100)
       print('Accuracy on Test data : ', knn_test_data_accuracy)


       
       model = GaussianNB()                                                                    
       model.fit(X_train, Y_train)
       X_train_prediction = model.predict(X_train)
       g_training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
       g_training_data_accuracy="{:.2f}".format(g_training_data_accuracy*100)
       print('Accuracy on Training data : ', g_training_data_accuracy)
       X_test_prediction = model.predict(X_test)
       g_test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
       g_test_data_accuracy="{:.2f}".format(g_test_data_accuracy*100)
       print('Accuracy on Test data : ', g_test_data_accuracy)

       


       """DIABETIC......."""


       nop=int(request.form.get('nop'))
       glucose=int(request.form.get('glucose'))
       st=int(request.form.get('st'))
       insulin=int(request.form.get('insulin'))
       bmi=float(request.form.get('bmi'))
       dpf=float(request.form.get('dpf'))
       bp=int(request.form.get('bp'))

       diabetes_data=pd.read_csv('diabetes.csv')
       diabetes_data['Outcome'].value_counts()
       A = diabetes_data.drop(columns='Outcome', axis=1)
       B = diabetes_data['Outcome']

       
       A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.20, stratify=B, random_state=2)
       print(A.shape, A_train.shape, A_test.shape)

       
       model1 = LogisticRegression()
       model1.fit(A_train, B_train)
       A_train_prediction = model1.predict(A_train)
       lr_training_data_accuracy1 = accuracy_score(A_train_prediction, B_train)
       lr_training_data_accuracy1="{:.2f}".format(lr_training_data_accuracy1*100)
       print('Accuracy on Training data : ', lr_training_data_accuracy1)
       A_test_prediction = model1.predict(A_test)
       lr_test_data_accuracy1 = accuracy_score(A_test_prediction, B_test)
       lr_test_data_accuracy1="{:.2f}".format(lr_test_data_accuracy1*100)
       print('Accuracy on Test data : ', lr_test_data_accuracy1)


       model1 = KNeighborsClassifier(n_neighbors=5)                                                                    
       model1.fit(A_train, B_train)
       A_train_prediction = model1.predict(A_train)
       knn_training_data_accuracy1 = accuracy_score(A_train_prediction, B_train)
       knn_training_data_accuracy1="{:.2f}".format(knn_training_data_accuracy1*100)
       print('Accuracy on Training data : ', knn_training_data_accuracy1)
       A_test_prediction = model1.predict(A_test)
       knn_test_data_accuracy1 = accuracy_score(A_test_prediction, B_test)
       knn_test_data_accuracy1="{:.2f}".format(knn_test_data_accuracy1*100)
       print('Accuracy on Test data : ', knn_test_data_accuracy1)


       
       model1 = GaussianNB()                                                                    
       model1.fit(A_train, B_train)
       A_train_prediction = model1.predict(A_train)
       g_training_data_accuracy1 = accuracy_score(A_train_prediction, B_train)
       g_training_data_accuracy1="{:.2f}".format(g_training_data_accuracy1*100)
       print('Accuracy on Training data : ', g_training_data_accuracy1)
       A_test_prediction = model1.predict(A_test)
       g_test_data_accuracy1 = accuracy_score(A_test_prediction, B_test)
       g_test_data_accuracy1="{:.2f}".format(g_test_data_accuracy1*100)
       print('Accuracy on Test data : ', g_test_data_accuracy1)



       input_data = (age, sex, cp, bp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

# change the input data to a numpy array
       input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
       input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

       prediction = model.predict(input_data_reshaped)


       input_data1 = (nop,glucose,bp/2,st,insulin,bmi,dpf,age)

# change the input data to a numpy array
       input_data_as_numpy_array1= np.asarray(input_data1)

# reshape the numpy array as we are predicting for only on instance
       input_data_reshaped1 = input_data_as_numpy_array1.reshape(1,-1)

       prediction1 = model1.predict(input_data_reshaped1)



       '''print(age,thal,thalach,sep="-----")
       print(prediction)'''

       if (prediction[0]== 1):
            pred="YES"
       else:
            pred="NO"


       if (prediction1[0]== 1):
            pred1="YES"
       else:
            pred1="NO"

       with open("C:\\xampp\\htdocs\\project\\result.csv","a",newline="") as File:
              writer=csv.writer(File)
              writer.writerow([name,age,pred,pred1])
       File.close()

       return render_template('home.html',res_lr=pred,acc_train_lr=lr_training_data_accuracy,acc_test_lr=lr_test_data_accuracy,res_knn=pred,acc_train_knn=knn_training_data_accuracy,acc_test_knn=knn_test_data_accuracy,res_g=pred,acc_train_g=g_training_data_accuracy,acc_test_g=g_test_data_accuracy,res_lr1=pred1,acc_train_lr1=lr_training_data_accuracy1,acc_test_lr1=lr_test_data_accuracy1,res_knn1=pred1,acc_train_knn1=knn_training_data_accuracy1,acc_test_knn1=knn_test_data_accuracy1,res_g1=pred1,acc_train_g1=g_training_data_accuracy1,acc_test_g1=g_test_data_accuracy1)
   else:
       return render_template('home.html')
if __name__ == '__main__':
   app.run(debug = True)


   
