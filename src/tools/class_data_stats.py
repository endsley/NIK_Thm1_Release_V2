#!/usr/bin/env python

import numpy as np
import sys
import sklearn.metrics
from sklearn import preprocessing



class class_data_stats():
	def __init__(self, X, Y):	
		self.X = X
		self.Y = Y

		self.l = np.unique(Y)
		self.c = len(self.l)
		self.X_list = {}
		self.Y_list = {}
	
		for i in self.l:
			indices = np.where(Y == i)[0]
			self.X_list[i] = {}
			self.Y_list[i] = {}

			self.X_list[i]['X'] = X[indices, :]
			self.Y_list[i]['Y'] = Y[indices]



	def separate_data_by_class(self):
		XL = []
		for e, i in enumerate(self.l):
			indices = np.where(self.Y == i)[0]
			XL.append(self.X[indices, :])

		return XL

	def get_class_info(self):
		D = np.zeros((self.c, self.c))
		max_std = 0
		for e, i in enumerate(self.l):
			indices = np.where(self.Y == i)[0]
			self.X_list[i]['shape'] = self.X[indices, :].shape
			pDM = self.X_list[i]['pairwise_distance'] = sklearn.metrics.pairwise.pairwise_distances(self.X_list[i]['X'])
			kq = np.where(pDM == 0)
			vv = np.vstack(kq).T
			kq2 = kq[0] - kq[1]
			ids = np.where(kq2 != 0)[0]
			repeated_sample_pairs = vv[ids,:]
			num_of_repeated_sample_pairs = len(np.unique(repeated_sample_pairs[:,0]))


			self.X_list[i]['pairwise_distance_std'] = np.std(self.X_list[i]['pairwise_distance'])
			self.X_list[i]['pairwise_distance_max'] = np.max(self.X_list[i]['pairwise_distance'])
			self.X_list[i]['repeated_sample_pairs'] = repeated_sample_pairs

			if self.X_list[i]['pairwise_distance_std'] > max_std:
				max_std = self.X_list[i]['pairwise_distance_std']

			#print('Class %d'%i)
			#print('\t Number of sample with repeated value : %d, percetage of total %.3f'%(num_of_repeated_sample_pairs, num_of_repeated_sample_pairs/self.X_list[i]['shape'][0]))
			#print('\t data size : ', self.X_list[i]['shape'])
			#print('\t distance std : %.3f'% self.X_list[i]['pairwise_distance_std'])
			#print('\t distance max : %.3f'% self.X_list[i]['pairwise_distance_max'])
			D[e,e] = self.X_list[i]['pairwise_distance_max']

		for a, i in enumerate(self.l):
			for b, j in enumerate(self.l):
				if i != j:
					indices_i = np.where(self.Y == i)[0]
					indices_j = np.where(self.Y == j)[0]
	
					pd = sklearn.metrics.pairwise.pairwise_distances(self.X_list[i]['X'], self.X_list[j]['X'])

					# find repeated samples between classes
					kq = np.where(pd== 0)
					vv = np.vstack(kq).T
					kq2 = kq[0] - kq[1]
					ids = np.where(kq2 != 0)[0]
					repeated_sample_pairs = vv[ids,:]
					num_of_repeated_sample_pairs = len(np.unique(repeated_sample_pairs[:,0]))


					pd_min = np.min(pd)
					D[a,b] = pd_min
		print('Class max/min pairwise distance')
		print('\t' + str(D).replace('\n', '\n\t'))
		print('Contradictory Samples :')
		print('\t' + str(repeated_sample_pairs).replace('\n', '\n\t'))

		return max_std

	# labels unsorted into sorted label of same class next to eachother
	def rearrange_sample_to_same_class(self, X,Y):
		l = np.unique(Y)
		newX = np.empty((0, X.shape[1]))
		newY = np.empty((0))
	
		for i in l:
			indices = np.where(Y == i)[0]
			newX = np.vstack((newX, X[indices, :]))
			newY = np.hstack((newY, Y[indices]))
	
		return [newX, newY]

if __name__ == "__main__":
	np.set_printoptions(precision=4)
	np.set_printoptions(threshold=sys.maxsize)
	np.set_printoptions(linewidth=300)
	np.set_printoptions(suppress=True)


	data_name = 'cancer'
	
	X = np.loadtxt('data/' + data_name + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt('data/' + data_name + '_label.csv', delimiter=',', dtype=np.int32)			
	X = preprocessing.scale(X)

	CS = class_data_stats(X,Y)
	CS.get_class_info()

	np.savetxt(data_name + '.csv', X, delimiter=',', fmt='%.4f') 
	np.savetxt(data_name + '_label.csv', Y, delimiter=',', fmt='%d') 
