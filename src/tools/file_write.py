

from .path_tools import *
import numpy as np

def write_train_results(ʆ, fname):
	fin = open(fname, 'w')
	fin.write(ʆ + '\n')
	fin.close()


def file_readline(fname):
	fin = open(fname, 'r')
	lines = fin.readlines()
	fin.close()
	return lines

def list_num_to_mean_std(l):
	L = np.array(l)
	m = np.mean(L)
	s = np.std(L)

	txt = '%.2f±%.2f'%(m,s)
	return txt

def save_result(pth, content, print_result=False):
	fin = open(pth, 'a')
	fin.write('%s\n'%content)
	fin.close()
	if print_result: print(content)


