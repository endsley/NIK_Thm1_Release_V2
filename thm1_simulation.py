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
from sklearn.preprocessing import LabelEncoder

from src.kchain.kchain import *
from src.tools.one_hot_encoder import *
from src.tools.path_tools import *
from src.tests.run_10_fold import *


np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)


class wumasoomi():
	def __init__(self, X, Y, data_name, smallest_σ=0.0001, number_of_σ_to_test=10, HSIC_exit_threshold=0.95, use_kernel_trick=True):	
		self.db = db = {}
		self.db['X'] = X
		self.db['Y'] = LabelEncoder().fit_transform(Y)
		self.db['Yₒ'] = one_hot_encoding(db['Y'])
		self.db['n'] = db['Y'].shape[0]
		self.db['Ł_目'] = np.unique(db['Y'])
		self.db['c'] = len(self.db['Ł_目'])
		self.db['RFF_ð'] = 200						# width of RFF
		self.db['max_layer'] = 30
		self.db['data_name'] = data_name
		self.db['smallest_σ'] = smallest_σ									# when looping through σ, smallest value
		self.db['number_of_σ_to_test'] = number_of_σ_to_test				# when looping through σ, list size
		self.db['HSIC_exit_threshold'] = HSIC_exit_threshold
		self.db['kchain'] = kchain(self.db)
		self.db['use_kernel_trick'] = True


	def fit(self, ᘐ):
		self.db['kchain'].fit(ᘐ)
		return self.db['kchain']
			

if __name__ == "__main__":

# Run 10 Fold, uncomment the experiment you want to run
#	run_10_fold('wine', wumasoomi, smallest_σ=0.05, number_of_σ_to_test=20)
#	run_10_fold('cancer', wumasoomi, smallest_σ=0.05, number_of_σ_to_test=20)
#	run_10_fold('random', wumasoomi, smallest_σ=0.05, number_of_σ_to_test=20)
#	run_10_fold('adversarial', wumasoomi, smallest_σ=0.001, number_of_σ_to_test=20)
#	run_10_fold('spiral', wumasoomi, smallest_σ=0.05, number_of_σ_to_test=20)
#	run_10_fold('car', wumasoomi, smallest_σ=0.05, number_of_σ_to_test=20)
#	run_10_fold('divorce', wumasoomi, smallest_σ=0.1, number_of_σ_to_test=40, use_kernel_trick=True)
#	run_10_fold('face', wumasoomi, smallest_σ=0.05, number_of_σ_to_test=20)
	run_10_fold('cifar10', wumasoomi, smallest_σ=0.05, number_of_σ_to_test=20)


	#evaluate_10_fold_result('./results/' + data_name + '/10Fold_summary.txt')
	#CS = class_data_stats(X, Y)
	#CS.get_class_info()

	#wm = wumasoomi(X,Y, data_name)
	#wm.train()





