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



def main():
	dataset = np.loadtxt('spambase.data', delimiter=',')

	knn = neighbors.KNeighborsClassifier()
	svm = SVC()
	d_tree = tree.DecisionTreeClassifier()

	X = dataset[:,0:52]# 57 ger konstiga tal, why??
	y = dataset[:, -1]
	k = 10

	kf = KFold(n_splits=k, random_state=None, shuffle=False) #ändra till shuff=true sen
	kf.get_n_splits(X)


	#print(kf)
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		get_score(knn, X_train, X_test, y_train, y_test, knn_accuracy, knn_F_score, knn_time)
		get_score(svm, X_train, X_test, y_train, y_test, svm_accuracy, svm_F_score, svm_time)
		get_score(d_tree, X_train, X_test, y_train, y_test, d_tree_accuracy, d_tree_F_score, d_tree_time)



	K_Fold_Result(knn_accuracy, svm_accuracy, d_tree_accuracy, k)	

	
def K_Fold_Result(model_1_meassurement, model_2_meassurement, model_3_meassurement, k_nr):
	knn_avg, svm_avg, d_tree_avg = 0.0, 0.0, 0.0

	print("Accuracy")
	print("Fold\t    " + "KNN\t\t\t" + "SVM\t\t" + "Decision Tree")
	for i in range(len(model_1_meassurement)):
	    knn_avg += model_1_meassurement[i]
	    svm_avg += model_2_meassurement[i]
	    d_tree_avg += model_3_meassurement[i]
	    print(str(i + 1) + "\t    " + format(model_1_meassurement[i],".3f") + "\t\t" + format(model_2_meassurement[i],".3f") + "\t\t" + format(model_3_meassurement[i],".3f"))
	print("---------------------------------------------------------------")
	knn_avg = knn_avg/k_nr
	svm_avg = svm_avg/k_nr
	d_tree_avg = d_tree_avg/k_nr
	print("Avg" + "\t    " + format(knn_avg,".3f") + "\t\t" + format(svm_avg,".3f") + "\t\t" + format(d_tree_avg,".2f"))
	#print('stdev' + "\t    " + ("%.3f" %np.std(knn_accuracy)) + "\t\t" + ("%.3f" %np.std(svm_accuracy)) + "\t\t" + ("%.3f" %np.std(d_tree_accuracy)))
	#knn_avg, svm_avg, d_tree_avg = 0, 0, 0
	print("---------------------------------------------------------------")



'''
Trains, tests the model, calculates the training time, gets the accuracy and the f1 score.
'''
def get_score(model, X_train, X_test, y_train, y_test, model_accuracy, model_f_score, model_time):
		t0=time()
		model.fit(X_train, y_train)
		model_time.append(round(time()-t0, 2))

		y_predicted = model.predict(X_test)

		model_accuracy.append(model.score(X_test, y_test))
		model_f_score.append(f1_score(y_test, y_predicted, average='macro'))
		
		#return model.score(X_test, y_test)

if __name__ =="__main__":
	main()



'''
		y_predicted = model.predict(X_test)

		model_accuracy.append(accuracy_score(y_test, y_predicted))
		model_f_score.append(f1_score(y_test, y_predicted, average='macro'))
		
		return model.score(X_test, y_test)
'''