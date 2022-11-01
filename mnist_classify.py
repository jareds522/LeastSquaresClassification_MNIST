'''
THIS FILE IS FROM JUPYTER, SO IT IS NOT A WORKING STANDALONE CLASSIFICATION SCRIPT
THE PURPOSE OF THIS SCRIPT IS TO LOAD IN AND PREPROCESS THE DATASET
'''
from keras.datasets import mnist
import numpy as np
from sklearn import preprocessing as pp
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import copy


#value is what we are checking the output against (in range 0-9)
#returns the classifier's prediction 
def binaryClassifier(x_train, y_train, x_test, value):
    #Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    #Convert y outputs to binary (required for LogisticRegression() method)
    yi_train = copy.copy(y_train)
    yi_train[yi_train != value] = -1
    yi_train[yi_train == value] = 1
    yi_train[yi_train == -1] = 0
    
    model = LogisticRegression(max_iter = 400)
    model.fit(x_train, yi_train)
    
    #In this method, yi_predict represents beta*x + alpha, our solution to the LS
    #In this method, model.coef represents beta and model.intercept represents alpha
    y_predict = model.predict(x_test)
    
    #TODO: convert y_predict to +1 and -1?
    return y_predict, model.coef_, model.intercept_
    
#ONE VERSUS ALL CLASSIFIER
#perform the binary classification 10 times for i=0, 1 ... 9
def oneVersusAll(train_x, train_y, test_x, test_y):
    classifier_output = np.zeros((len(test_y), 2))
    for i in range(10):
        print("Solving for k = ", i)
        yi_test = copy.copy(test_y)
        yi_test[yi_test != i] = -1
        yi_test[yi_test == i] = 1
        yi_test[yi_test == -1] = 0
        
        yi_predict, beta, alpha = binaryClassifier(train_x, train_y, test_x, i)
        
        #classifier_output will store the highest "confidence" in the first index and the value i={0,1...,9} in the second
        for n in range(len(classifier_output)):
            if np.matmul(beta, test_x[n]) + alpha > classifier_output[n][0]:
                classifier_output[n][0] = np.matmul(beta, test_x[n]) + alpha
                classifier_output[n][1] = i
        
    return classifier_output

#load in .mat data
(train_x, train_y), (test_x, test_y) = mnist.load_data()

#Convert the data to float
train_x = np.array(train_x).astype(np.float64)
train_y = np.array(train_y).astype(np.float64)
test_x = np.array(test_x).astype(np.float64)
test_y = np.array(test_y).astype(np.float64)


#Convert data from 28x28 matrices to arrays of length 784 & normalize
train_x = pp.normalize(train_x.reshape(60000, 784))
test_x = pp.normalize(test_x.reshape(10000, 784))

#Perform 1 vs all classification using binary classification as building block
y_predict = oneVersusAll(train_x, train_y, test_x, test_y)

#Perform 1 vs 1 classification using binary classification as building block
