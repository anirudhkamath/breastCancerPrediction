import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as mp
from sklearn import tree, metrics, model_selection

dataFrame = pd.read_csv('dataset.csv', names = ['ID','clumpThickness', 'cellSize', 'cellShapeUniformity', 
						'marginalAdhesion', 'singleEpithelialCellSize', 'bareNuclei', 'blandChromatin', 'normalNucleoli', 'Mitoses',
						'Class'])
print(dataFrame.head())
print(dataFrame.info())

#all values are non-null objects, but we need it in integer format for performing analysis on it for training the decision tree
dataFrame['Class'],class_values = pd.factorize(dataFrame['Class'])
print(class_values) #to see what values of class are there. 2 represents benign, 4 represents malignant
print(dataFrame['Class'].unique())

#bareNuclei only has 683 values instead of 699. So attempting to smooth the values by replacing null values by mean of the attribute
#print(dataFrame.isnull().values.any())

temp=0
temp = dataFrame['bareNuclei'].sum()
temp = int(temp/683) #to get mean of values
temp = float(temp)

for i in dataFrame['bareNuclei']:
	if i==np.NaN:
		i=temp

dataFrame['bareNuclei'].fillna(temp, inplace=True) #smoothing all missing values by mean
#print(dataFrame['bareNuclei'])

#next, we need to select prediction variables (that will help in classification), and the labels given to those variable choices.
x = dataFrame.iloc[:,:-1] #all attributes except class
y = dataFrame.iloc[:,-1] #only last attribute

#every supervised predictor has training data and testing data. let us split our data into those two fragments.
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.3,random_state=0) #test data is 30% of all data

print(x_test)

dTree = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None, random_state=0) #need to find optimum value for max_depth of tree
dTree.fit(x_train, y_train)

#now to get the predicted value for test variables
y_pred = dTree.predict(x_test)

#now to check performance metrics and see how many mismatches
count_misclassified = (y_test!=y_pred).sum()
print("Misclassified samples: {}".format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))