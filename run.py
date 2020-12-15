# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd# Importing the dataset

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn import tree

from sklearn.model_selection import KFold

#from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from time import time
import statistics

score_knn = []
#KNN
knn_accuracy = []
knn_F_score = []
knn_time = []
#SVM
svm_accuracy = []
svm_F_score = []
svm_time = []
#Decision tree
d_tree_accuracy = []
d_tree_F_score = []
d_tree_time = []

line = "---------------------------------------------------------------"

def main():
	dataset = np.loadtxt('spambase.data', delimiter=',')

	knn = neighbors.KNeighborsClassifier()
	svm = SVC()
	d_tree = tree.DecisionTreeClassifier()

	X = dataset[:,0:52]# 57 ger konstiga tal, why??
	y = dataset[:, -1]
	k = 10

	kf = KFold(n_splits=k, random_state=None, shuffle=True) 
	kf.get_n_splits(X)


	#print(kf)
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		get_scores(knn, X_train, X_test, y_train, y_test, knn_accuracy, knn_F_score, knn_time)
		get_scores(svm, X_train, X_test, y_train, y_test, svm_accuracy, svm_F_score, svm_time)
		get_scores(d_tree, X_train, X_test, y_train, y_test, d_tree_accuracy, d_tree_F_score, d_tree_time)



	get_results(knn_accuracy, svm_accuracy, d_tree_accuracy, k, "Accuracy")	
	get_results(knn_F_score, svm_F_score, d_tree_F_score, k, "F-measure")	
	get_results(knn_time, svm_time, d_tree_time, k, "Training time")

	print("Friedman's test: Accuracy")
	f_test(knn_accuracy, svm_accuracy, d_tree_accuracy, k)
	print("Friedman's test: F-measure")
	f_test(knn_F_score, svm_F_score, d_tree_F_score, k)	
	print("Friedman's test: Time time")
	f_test(knn_time, svm_time, d_tree_time, k)		




def f_test(model_1_meassurement, model_2_meassurement, model_3_meassurement, k_nr):
	knn_avg, svm_avg, d_tree_avg = 0.0, 0.0, 0.0
	knn, s, d = 0, 0, 0

	print("Dataset\t    " + "KNN\t\t\t" + "SVM\t\t" + "Decision Tree")
	for i in range(len(model_1_meassurement)):
		if model_1_meassurement[i] > model_2_meassurement[i] and model_1_meassurement[i] > model_3_meassurement[i]:
			knn_avg += 1
			knn = 1
			if model_2_meassurement[i] > model_3_meassurement[i]:
				svm_avg += 2
				d_tree_avg += 3
				s, d = 2, 3
			else:
				d_tree_avg += 2
				svm_avg += 3
				d, s = 2, 3
		elif model_2_meassurement[i] > model_1_meassurement[i] and model_2_meassurement[i] > model_3_meassurement[i]:
			svm_avg += 1
			s = 1
			if model_1_meassurement[i] > model_3_meassurement[i]:
				knn_avg += 2
				d_tree_avg += 3
				knn, d = 2, 3
			else:
				d_tree_avg += 2
				knn_avg += 3
				d, knn = 2, 3
		elif model_3_meassurement[i] > model_1_meassurement[i] and model_3_meassurement[i] > model_2_meassurement[i]:
			d_tree_avg += 1
			d = 1
			if model_1_meassurement[i] > model_2_meassurement[i]:
				knn_avg += 2
				svm_avg += 3
				knn, s = 2, 3
			else:
				svm_avg += 2
				knn_avg += 3
				s, knn = 2, 3

		print((str(i + 1) + "\t    " + format(model_1_meassurement[i],".3f") + "(" + str(knn) + ")" + "\t\t" + 
			format(model_2_meassurement[i],".3f") + "(" + str(s) + ")" + "\t" + format(model_3_meassurement[i],".3f")+ "(" + str(d) + ")"))
	
	knn_avg = knn_avg/k_nr
	svm_avg = svm_avg/k_nr
	d_tree_avg = d_tree_avg/k_nr

	print(line)
	print("Avg" + "\t    " + format(knn_avg, ".1f") + "\t\t\t" + format(svm_avg, ".1f") + "\t\t" + format(d_tree_avg, "1f"))
	print(line)

	'''
	Determine whether the average ranks as a whole display significant differences 
	on the 0.05 alpha level and, if so, use the Nemeyi test to calculate critical 
	difference in order to determine which algorithms perform significantly different from each other
	'''

'''
Models: KNN, SVM, d_tree
'''
def get_results(model_1_meassurement, model_2_meassurement, model_3_meassurement, k_nr, measurement):
	knn_avg, svm_avg, d_tree_avg = 0.0, 0.0, 0.0

	print(measurement)
	print("Fold\t    " + "KNN\t\t\t" + "SVM\t\t" + "Decision Tree")
	for i in range(len(model_1_meassurement)):
	    print(str(i + 1) + "\t    " + format(model_1_meassurement[i],".3f") + "\t\t" + 
	    	format(model_2_meassurement[i],".3f") + "\t\t" + format(model_3_meassurement[i],".3f"))
	
	print(line)
	print("Avg" + "\t    " + format(statistics.mean(model_1_meassurement),".3f") + 
		"\t\t" + format(statistics.mean(model_2_meassurement),".3f") + "\t\t" + format(statistics.mean(model_3_meassurement),"3f"))
	print("Stdev" + "\t    " + format(np.std(model_1_meassurement), ".3f") + 
		"\t\t" + format(np.std(model_2_meassurement), ".3f") + "\t\t" + format(np.std(model_3_meassurement), ".3f"))
	print(line)



'''
Trains, tests the model, calculates the training time, gets the accuracy and the f1 score.
'''
def get_scores(model, X_train, X_test, y_train, y_test, model_accuracy, model_f_score, model_time):
		t0=time()
		model.fit(X_train, y_train)
		model_time.append(round(time()-t0, 2))

		y_predicted = model.predict(X_test)

		model_accuracy.append(model.score(X_test, y_test))
		model_f_score.append(f1_score(y_test, y_predicted, average='macro'))
		
		#return model.score(X_test, y_test)

if __name__ =="__main__":
	main()

