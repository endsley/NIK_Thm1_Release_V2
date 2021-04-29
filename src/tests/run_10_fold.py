
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import SVC
import time
import gc

from src.tools.path_tools import *
from src.tools.hsic import *
from src.tools.one_hot_encoder import *
from src.tools.file_write import *
from src.tools.distances import *
from src.tools.split_10_fold import *


#def compute_MSE_result(ӯ, Y):
#	l = np.unique(Y)
#
#	for e, i in enumerate(l):
#		indices = np.where(self.Y == i)[0]
#		mse = np.sum(np.linalg.norm(ӯ[indices,:] - np.mean(ӯ[indices,:], axis=0), axis=1))



def evaluate_10_fold_result(complete_summary_path):
	lines = file_readline(complete_summary_path)
	layer目 = []
	minσ目 = []
	trainAcc目 = []
	testAcc目 = []
	time目 = []
	Ҥ1目 = []				# HSIC list on Training
	Ҥ2目 = []				# HSIC list on Test
	mse目 = []				# HSIC list on Test
	ce目 = []				# HSIC list on Test
	csr目 = []				# HSIC list on Test
	T目 = []				# HSIC list on Test

	dataset_name = os.path.basename(go_up_1_directory(complete_summary_path))

	for l in lines:
		if l.find('run') == 0:
			items = l.split(',')

			minσ目.append(float(items[1].split(':')[1].strip()))
			layer目.append(float(items[2].split(':')[1].strip()))
			trainAcc目.append(float(items[3].split(':')[1].strip()))
			testAcc目.append(float(items[4].split(':')[1].strip()))
			time目.append(float(items[5].split(':')[1].strip()))
			Ҥ1目.append(float(items[6].split(':')[1].strip()))
			Ҥ2目.append(float(items[7].split(':')[1].strip()))
			mse目.append(float(items[8].split(':')[1].strip()))
			ce目.append(float(items[9].split(':')[1].strip()))
			csr目.append(float(items[10].split(':')[1].strip()))
			T目.append(float(items[11].split(':')[1].strip()))


	# Results for AIStats paper
	txt = '\n%-13s%-13s%-13s%-13s%-13s%-13s%-13s%-13s%-13s%-13s%-13s\n'%('data', 'smallest σ', 'layer #', 'Train Acc', 'Test Acc', 'Time', 'H', 'mse', 'ce','C', 'T')
	txt += '%-13s'%(dataset_name)
	txt += '%-13.2f'%(np.min(minσ目))
	txt += '%-13s'%(list_num_to_mean_std(layer目))
	txt += '%-13s'%(list_num_to_mean_std(trainAcc目))
	txt += '%-13s'%(list_num_to_mean_std(testAcc目))
	txt += '%-13s'%(list_num_to_mean_std(time目))
	txt += '%-13s'%(list_num_to_mean_std(Ҥ1目))
	txt += '%-13s'%(list_num_to_mean_std(mse目))
	txt += '%-13s'%(list_num_to_mean_std(ce目))
	txt += '%-13s'%(list_num_to_mean_std(csr目))
	txt += '%-13s'%(list_num_to_mean_std(T目))

	save_result(complete_summary_path, txt, print_result=True)

#	# Results for the workshop paper
#	txt = '\n%-18s%-18s%-18s%-18s%-18s%-18s\n'%('data', 'layer #', 'smallest σ', 'Train Acc', 'Test Acc', 'Time')
#	txt += '%-18s'%(dataset_name)
#	txt += '%-18s'%(list_num_to_mean_std(layer目))
#	txt += '%-18s'%(list_num_to_mean_std(minσ目))
#	txt += '%-18s'%(list_num_to_mean_std(trainAcc目))
#	txt += '%-18s'%(list_num_to_mean_std(testAcc目))
#	txt += '%-18s'%(list_num_to_mean_std(time目))
#
#	save_result(complete_summary_path, txt)

def evaluate_1_fold_result(ᘐ, pth, X, Y, X_test, Y_test, kchain):
	Yₒ = one_hot_encoding(Y)
	Yₒ_test = one_hot_encoding(Y_test)
	complete_summary_path = go_up_1_directory(pth) + '/10Fold_summary.txt'
	

	ӯ = kchain.get_network_output(X)
	σ = kchain.LM.σᵦ
	[Ҥ1,K] = ℍ(ӯ,Yₒ, σ, Kᵪ_type='Gaussian', Kᵧ_type='linear')


	neigh = KNeighborsClassifier()
	#neigh = SVC(gamma='auto')
	out_allocation = neigh.fit(ӯ, Y).predict(ӯ)
	acc1 = accuracy_score(Y, out_allocation)
	ӯ2 = kchain.get_network_output(X_test)
	out_allocation2 = neigh.predict(ӯ2)
	acc2 = accuracy_score(Y_test, out_allocation2)
	[Ҥ2,K] = ℍ(ӯ2,Yₒ_test, σ, Kᵪ_type='Gaussian', Kᵧ_type='linear')

	# compare to pure KNN
	out_allocation3 = neigh.fit(X, Y).predict(X)
	acc3 = accuracy_score(Y, out_allocation3)

	out_allocation4 = neigh.predict(X_test)
	acc4 = accuracy_score(Y_test, out_allocation4)

	minσ = np.min(kchain.LM.get_σ_list())
	num_layer = len(kchain.LM.ℓ目)


	mse = MSE(ӯ,Y)
	ce = Get_Cross_Entropy(ӯ,Y, Yₒ)
	sR = scatter_ratio(ӯ, Yₒ)
	csr = get_CSR(ӯ,Y, kchain.LM.σᵦ)

	#fold_summary = 'Ҥ1 : %.3f, Ҥ2 : %.3f, Acc Train : %.3f , Acc Test : %.3f, KNN Acc Train : %.3f, KNN Acc Test : %.3f, time : %.3f, '%(Ҥ1, Ҥ2, acc1, acc2, acc3, acc4, kchain.db['Δt'])
	#fold_summary += 'min σ : %.3f, num layer: %d'%(minσ, num_layer)
	#final_out = 'layer : %d, min σ : %.2f, Acc Train : %.2f, Acc Test : %.2f, time : %.2f'%(num_layer, minσ, acc1, acc2, kchain.db['Δt'])

	fold_summary = 'run:%d, σ: %.2f, L: %d, TrainAcc: %.2f, TestAcc: %.2f, Time: %.2f, '%(ᘐ, minσ, num_layer, acc1, acc2, kchain.db['Δt'])
	fold_summary += 'Ҥ1: %.2f, Ҥ2:%.2f, mse : %.2f, ce : %.2f, csr : %.2f, T : %.2f'%(Ҥ1, Ҥ2, mse, ce, csr, sR)


	#, Ҥ2 : %.3f, Acc Train : %.3f , Acc Test : %.3f, KNN Acc Train : %.3f, KNN Acc Test : %.3f, time : %.3f'%(Ҥ1, Ҥ2, acc1, acc2, acc3, acc4, kchain.db['Δt'])
	save_result(pth + 'layer_output_summary.txt', fold_summary, print_result=True)
	save_result(complete_summary_path, fold_summary)




def run_10_fold(data_name, prog, 
				run_list=None, 
				smallest_σ=0.0001, 
				number_of_σ_to_test=10, 
				HSIC_exit_threshold=0.95, 
				use_kernel_trick=True):
	gen_10_fold_data(data_name, data_path='./data/')	

	pth_name = data_name + '/' + data_name + '_'
	ensure_path_exists('./results')
	ensure_path_exists('./results/' + data_name)
	delete_file('./results/' + data_name + '/10Fold_summary.txt')


	if run_list is None: run_list = range(1,11)
	for ᘐ in run_list:
		file_name = pth_name + str(ᘐ)
		pth = './results/' + file_name + '/'
		initialize_empty_folder('./results/' + file_name + '/')

		# load and preprocess data
		X = np.loadtxt('data/' + file_name + '.csv', delimiter=',', dtype=np.float64)			
		Y = np.loadtxt('data/' + file_name + '_label.csv', delimiter=',', dtype=np.int32)			
		X_test = np.loadtxt('data/' + file_name + '_test.csv', delimiter=',', dtype=np.float64)			
		Y_test = np.loadtxt('data/' + file_name + '_label_test.csv', delimiter=',', dtype=np.int32)			

		#X = np.vstack((X,X,X, X))			# if there are too few samples, try repeating data and add some noise
		#Y = np.hstack((Y,Y,Y, Y))
		
		X = X + 0.02*np.random.randn(X.shape[0], X.shape[1])
		X = preprocessing.scale(X)	
		X_test = preprocessing.scale(X_test)

		# run program
		start_time = time.perf_counter()
		wm = prog(X,Y, file_name, 
					smallest_σ=smallest_σ, 
					number_of_σ_to_test=number_of_σ_to_test, 
					HSIC_exit_threshold=HSIC_exit_threshold,
					use_kernel_trick=use_kernel_trick)

		kchain = wm.fit(ᘐ)
		kchain.db['Δt'] = time.perf_counter() - start_time

		evaluate_1_fold_result(ᘐ, pth, X, Y, X_test, Y_test, kchain)

		del wm
		del kchain
		del X
		del Y
		del X_test
		del Y_test
		gc.collect()

	evaluate_10_fold_result('./results/' + data_name + '/10Fold_summary.txt')
	#print('Running %s, Using Kernel Trick %r, HSIC_exit_threshold %0.3f, smallest_σ %.3f'%(data_name, use_kernel_trick, HSIC_exit_threshold, smallest_σ))



