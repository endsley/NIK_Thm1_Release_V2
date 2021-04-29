#!/usr/bin/env python	

import numpy as np	
from sklearn.model_selection import KFold
import os
	
def gen_10_fold_data(data_name, data_path='./data/'):

	xpath = data_path + data_name

	X = np.loadtxt(xpath + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt(xpath + '_label.csv', delimiter=',', dtype=np.int32)			

	fold_path = xpath + '/'

	if os.path.exists(fold_path): 
		txt = input("There's already a 10 fold data for %s, generate a new set? (y/N)"%data_name)
		if txt == 'y':
			pass
		else:
			return
	else:
		os.mkdir(fold_path)

	if os.path.exists(fold_path): 
		pass
	else:
		os.mkdir(fold_path)


	kf = KFold(n_splits=10, shuffle=True)
	kf.get_n_splits(X)
	loopObj = enumerate(kf.split(X))

	for count, data in loopObj:
		[train_index, test_index] = data

		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		
		np.savetxt( fold_path + data_name + '_' + str(count+1) + '.csv', X_train, delimiter=',', fmt='%.6f') 
		np.savetxt( fold_path + data_name + '_' + str(count+1) + '_label.csv', Y_train, delimiter=',', fmt='%d') 
		np.savetxt( fold_path + data_name + '_' + str(count+1) + '_test.csv', X_test, delimiter=',', fmt='%.6f') 
		np.savetxt( fold_path + data_name + '_' + str(count+1) + '_label_test.csv', Y_test, delimiter=',', fmt='%d') 


if __name__ == "__main__":
	gen_10_fold_data('cancer', data_path='./data/')
