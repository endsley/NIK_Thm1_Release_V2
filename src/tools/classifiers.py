
import numpy as np
from sklearn import svm
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def Allocation_2_Y(allocation):
	N = np.size(allocation)
	unique_elements = np.unique(allocation)
	num_of_classes = len(unique_elements)
	class_ids = np.arange(num_of_classes)

	i = 0
	Y = np.zeros(num_of_classes)
	for m in allocation:
		class_label = np.where(unique_elements == m)[0]
		a_row = np.zeros(num_of_classes)
		a_row[class_label] = 1
		Y = np.hstack((Y, a_row))

	Y = np.reshape(Y, (N+1,num_of_classes))
	Y = np.delete(Y, 0, 0)

	return Y


def Get_Cross_Entropy(X,Y):		
	clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, Y)
	Yₒ = Allocation_2_Y(Y)
	ŷ = -clf.predict_log_proba(X)
	n = X.shape[0]

	CE = np.sum(Yₒ*ŷ)/n
	return CE

def use_svm(X,Y,k='rbf', K=None):	
	svm_object = svm.SVC(kernel=k)

	if K is None:
		svm_object.fit(X, Y)
		out_allocation = svm_object.predict(X)
	else:
		svm_object.fit(K, Y)
		out_allocation = svm_object.predict(K)

	#nmi = normalized_mutual_info_score(out_allocation, Y)
	acc = accuracy_score(out_allocation, Y)

	return [out_allocation, acc, svm_object]


def apply_svm(X,Y, svm_obj):
	out_allocation = svm_obj.predict(X)
	#nmi = normalized_mutual_info_score(out_allocation, Y)
	acc = accuracy_score(out_allocation, Y)
	return acc


