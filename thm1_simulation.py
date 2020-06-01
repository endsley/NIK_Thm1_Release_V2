#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

import sys
import matplotlib
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import socket
from sklearn import preprocessing

from src.kchain.kchain import *
from src.tools.one_hot_encoder import *
from src.tools.path_tools import *

np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)


class thm_simulation():
	def __init__(self, X, Y, data_name):	#	X=data, Y=label, q=reduced dimension
		#	automated variables
		self.db = {}

		self.db['Ⲭ'] = X
		#self.db['X'] = preprocessing.scale(X)
		self.db['X'] = X

		self.db['Y'] = Y
		self.db['Yₒ'] = one_hot_encoding(Y)
		self.db['n'] = Y.shape[0]
		self.db['Ł_目'] = np.unique(Y)
		self.db['c'] = len(self.db['Ł_目'])
		self.db['RFF_ð'] = 10000	# width of RFF
		self.db['max_layer'] = 14
		self.db['kchain'] = kchain(self.db)
		self.db['data_name'] = data_name
		self.db['smallest_σ'] = 0.0000000001
		self.db['default_σᶩ'] = 0.15		

		ensure_path_exists('./results')
		ensure_path_exists('./results/' + data_name)

	def train(self):
		self.db['kchain'].study_pattern()
			

if __name__ == "__main__":
	#data_name = 'random'
	data_name = 'adversarial'

	X = np.loadtxt('data/' + data_name + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt('data/' + data_name + '_label.csv', delimiter=',', dtype=np.int32)			

	wm = thm_simulation(X,Y, data_name)
	wm.train()





