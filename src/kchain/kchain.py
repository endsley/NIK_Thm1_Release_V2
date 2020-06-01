

import sklearn.metrics
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pickle
import os


from src.tools.path_tools import *
from src.tools.opt_gaussian import *
from src.tools.hsic import *
from src.tools.plot_lib import *
from src.tools.merge_images import *
from src.tools.file_write import *

class kchain():
	def __init__(self, db):	
		self.db = db
		db['Ky'] = db['Yₒ'].dot(db['Yₒ'].T)
		self.Γ = double_center(db['Ky'])
		self.ΣΓᵴ = np.sum(np.ceil(self.Γ)*self.Γ)
		self.class_mask = {}
		self.Γ_classes = {}
		

		for class_id in db['Ł_目']:
			Yₒ = np.copy(db['Yₒ'])
			indices = np.where(db['Y'] != class_id)[0]
			Yₒ[indices, :] = 0
			self.class_mask[class_id] = Yₒ.dot(Yₒ.T)
			self.Γ_classes[class_id] = self.class_mask[class_id]*self.Γ

		self.inv_mask = np.absolute(db['Ky'] - 1)
		self.betwee_class_Γ = self.Γ*self.inv_mask

	def obtain_avg_direction(self, rᶩ, class_id):
		db = self.db
		#print('\tUsing mean of class %d'%class_id)

		indices = np.where(db['Y'] == class_id)[0]
		r_class_i = rᶩ[indices, :]

		# A
		r_class_i_μ = np.sum(r_class_i, axis=0)
		rn = np.linalg.norm(r_class_i_μ)
		ṝ = r_class_i_μ/rn

		## B
		#r_class_i_μ = np.mean(r_class_i, axis=0)
		#rn_2 = np.linalg.norm(r_class_i_μ)
		#ṝ_2 = r_class_i_μ/rn_2

		ṝ = np.reshape(ṝ,(db['RFF_ð'],1))
		return [ṝ, rn]

	def get_Upper_bound(self,rᶩ):
		K = rᶩ.dot(rᶩ.T)
		np.fill_diagonal(K, 0)	# make sure the rᵢ ⵐ rᴊ
		ε = np.max(K)
		U = ε*self.ΣΓᵴ
		return [U,ε]

	def get_Lower_bound(self, ε, σᶩ, rn, class_id, Kᵪ):
		db = self.db
		n = db['n']
	
		#Lˢᑊ = np.exp(-(2*((1+(n-1)*ε)) - 2)/(2*σᶩ*σᶩ*rn*rn))
		#ε = 0.01
		Lˢᑊ = np.exp(-(2*np.power((1+(n-1)*ε), 2) - 2)/(2*σᶩ*σᶩ*rn*rn))
		#Lˢᒾ = np.exp(-np.power(n*ε,2)/(σᶩ*σᶩ*rn*rn))
		#Lˢᶜ = np.exp(-(1-2*n*ε*(1+(n-1)*ε))/(2*σᶩ*σᶩ*rn*rn))

		restKᵪ = self.Γ*Kᵪ*np.absolute(self.class_mask[class_id] - 1)
		current_class = Lˢᑊ*self.Γ_classes[class_id]
		np.fill_diagonal(current_class, 0)	# make sure the rᵢ ⵐ rᴊ

		L = np.sum(restKᵪ + current_class)
		#import pdb; pdb.set_trace()
		return L

	def find_σᶩᐨᑊ(self, rᶩᐨᑊ, σᶩᐨᑊ, layer_id):
		db = self.db		
		c_id = np.mod((layer_id - 1),db['c'])
		c_id = 0

		pth = './results/' + db['data_name'] + '/' + str(layer_id) + '/'
		[ℍᶩᐨᑊ,Kᵪ_1st, Hⁿ] = ℍ(rᶩᐨᑊ, db['Yₒ'], σᶩᐨᑊ, Kᵪ_type='Gaussian', Kᵧ_type='linear')
		#ℍᶩᐨᑊ = ℍ(rᶩᐨᑊ, db['Yₒ'], σᶩᐨᑊ, Kᵪ_type='Gaussian', Kᵧ_type='linear')

		ℱᴀ = RBFSampler(gamma=1.0/(2*σᶩᐨᑊ*σᶩᐨᑊ), n_components=db['RFF_ð'], random_state=None)	
		rᶩ = ℱᴀ.fit_transform(rᶩᐨᑊ)
		[U,ε] = self.get_Upper_bound(rᶩ)

		[ṝ, rn] = self.obtain_avg_direction(rᶩ, db['Ł_目'][c_id])
		zᶩ = rᶩ.dot(ṝ)

		[ℍᶩ,Kᵪ, Hⁿ] = ℍ(zᶩ, db['Yₒ'], self.db['default_σᶩ'], Kᵪ_type='Gaussian', Kᵧ_type='linear')
		L = self.get_Lower_bound(ε, self.db['default_σᶩ'], rn, db['Ł_目'][c_id], Kᵪ)
		

		#σᶩ = opt_gaussian(zᶩ, db['Y'], σ_type='ℍ').result.x[0]	#σ_type='ℍ' or 'maxKseparation' print('σᶩ',σᶩ)
		σᶩ = db['default_σᶩ']
		[ℍᶩ_best_σᶩ,Kᵪ, Hⁿ] = ℍ(zᶩ, db['Yₒ'], σᶩ, Kᵪ_type='Gaussian', Kᵧ_type='linear')
	
		ℱⲃ = RBFSampler(gamma=1.0/(2*σᶩ*σᶩ), n_components=db['RFF_ð'], random_state=None)	
		rᶩᐩᑊ = ℱⲃ.fit_transform(zᶩ)
		#print(rᶩᐩᑊ.dot(rᶩᐩᑊ.T))

		#rᶩᐩᑊ_2 = TSNE(n_components=2).fit_transform(rᶩᐩᑊ)
		#pca = PCA(n_components=2)
		#rᶩᐩᑊ_2 = pca.fit_transform(rᶩᐩᑊ)
		#print(rᶩᐩᑊ_2.dot(rᶩᐩᑊ_2.T))
		#import pdb; pdb.set_trace()
	
		#print(σᶩᐨᑊ, ℍᶩᐨᑊ, ℍᶩ_best_σᶩ)
		return [ṝ, zᶩ, ℍᶩ, ℍᶩᐨᑊ, σᶩ, ℱᴀ, U, L, ℍᶩ_best_σᶩ, rᶩᐩᑊ, Hⁿ]

	def record_layer_internal_info(self, internal_info):
		db = self.db

		[zᶩ_vs_σᶩᐨᑊ, ℍᶩ_目, ℍᶩᐨᑊ_目, σ_目, pth, U_目, L_目] = internal_info

		#normalized_c = np.linalg.norm(zᶩ_vs_σᶩᐨᑊ,axis=0)
		normalized_c = np.max(zᶩ_vs_σᶩᐨᑊ,axis=0)
		normalized_zᶩ_vs_σᶩᐨᑊ = zᶩ_vs_σᶩᐨᑊ/normalized_c

		σ_目 = np.array(σ_目)
		ℍᶩ_目 = np.array(ℍᶩ_目)
		ℍᶩᐨᑊ_目 = np.array(ℍᶩᐨᑊ_目)

		ℍσ_trend = np.vstack((σ_目, ℍᶩ_目, ℍᶩᐨᑊ_目, U_目, L_目))
		np.savetxt(pth + 'trend_z_σ.csv', zᶩ_vs_σᶩᐨᑊ, delimiter=',', fmt='%.5f') 
		np.savetxt(pth + 'normalized_trend_z_σ.csv', normalized_zᶩ_vs_σᶩᐨᑊ, delimiter=',', fmt='%.5f') 
		np.savetxt(pth + 'Hσ_trend.csv', ℍσ_trend, delimiter=',', fmt='%.5f') 

	def record_layer_parameters(self, final_layer_info):
		[ṝ, zᶩ, ℍᶩ, ℍᶩᐨᑊ, σᶩ, σᶩᐨᑊ, ℱᴀ, pth] = final_layer_info

		Ł = ('ℍᶩᐨᑊ','ℍᶩ','σᶩ', 'σᶩᐨᑊ')
		ᘐ = (ℍᶩᐨᑊ,ℍᶩ,σᶩ, σᶩᐨᑊ)

		Ł_ʆ = ("%-10s\t%-10s\t%-10s\t%-10s"%Ł)
		ᘐ_ʆ = ("%-10.4f\t%-10.4f\t%-10.4f\t%-10.4f"%ᘐ)
		ʆ = Ł_ʆ + '\n' + ᘐ_ʆ
		write_train_results(ʆ, pth + 'summary.csv')


		np.savetxt(pth + 'zℓ.csv', zᶩ, delimiter=',', fmt='%.5f') 
		np.savetxt(pth + 'r_mean.csv', ṝ, delimiter=',', fmt='%.5f') 
		np.savetxt(pth + 'sigma.csv', [σᶩ] , delimiter=',', fmt='%.5f') 
		pickle.dump( ℱᴀ, open( pth + "RFF.pk", "wb" ) )

	def plot_internal_info(self, pth, count, layer_id, rᶩᐨᑊ, best_zᶩ, best_rᶩᐩᑊ):
		db = self.db
		K = best_rᶩᐩᑊ.dot(best_rᶩᐩᑊ.T)

		trend_z_σ = genfromtxt(pth + 'trend_z_σ.csv', delimiter=',')
		norm_trend_z_σ = genfromtxt(pth + 'normalized_trend_z_σ.csv', delimiter=',')

		Hσ_trend = genfromtxt(pth + 'Hσ_trend.csv', delimiter=',')


		# plot Z values	
		σs = Hσ_trend[0,:]
		plot_info = [trend_z_σ, 'Decreasing $\sigma$', 'All Sample Z values', 'Sample Z values as $\sigma$ Decrease', 'trend_z_σ.png', σs, pth]
		plot_heatMap(plot_info)

		plot_info = [norm_trend_z_σ, 'Decreasing $\sigma$', 'All Sample Z values', 'Sample $Z/|Z|_{\infty}}$ values as $\sigma$ Decrease', 'ntrend_z_σ.png', σs, pth]
		plot_heatMap(plot_info)


		#scatter(pth + 'orig_data.png', db['Ⲭ'], db['Y'], layer_id, 'Original Data' )
		scatter(pth + 'layer_input.png', rᶩᐨᑊ, db['Y'], layer_id, 'Layer Input' )
		#scatter(pth + 'data.png', rᶩᐨᑊ, db['Y'], layer_id,  'Data after center and scaled')
		scatter(pth + 'z_output.png', best_zᶩ, db['Y'], layer_id,  'network_Z_output')

		#scatter(pth + 'network_output.png', best_zᶩ, db['Y'], layer_id, 'Network output' )
		plot_info = [K, '', '', 'Network output Kernel', 'network_output.png', [], pth]
		plot_heatMap(plot_info)

		

		plot_info = [Hσ_trend, 'hsic_trend.png', pth]
		plot_hsic(plot_info)

		img_path_list = []
		img_path_list.append(pth + 'layer_input.png')
		img_path_list.append(pth + 'z_output.png')
		#img_path_list.append(pth + 'data.png')

		img_path_list.append(pth + 'network_output.png')

		img_path_list.append(pth + 'hsic_trend.png')
		img_path_list.append(pth + 'trend_z_σ.png')
		img_path_list.append(pth + 'ntrend_z_σ.png')
	
		imSize = Image.open(img_path_list[0]).size
		crop_window = (25, 0,imSize[0] - 10, imSize[1] - 0)
	
		pth2 = os.path.dirname(os.path.dirname(pth)) + '/'
		Imerger = img_merger(img_path_list, crop_window, horizontal=True)
		Imerger.save_img(pth2 + 'layer_' + str(count) + '_summary.png')
		Imerger.show_img()


	def study_pattern(self):
		db = self.db
		layer_id = 1
		ȋ = np.flip(np.linspace(db['smallest_σ'], np.sqrt(2), 20))
		last_best_ℍ = -100000							# best HSIC from last layer
		rᶩᐨᑊ = db['X']


		for repeat in range(db['max_layer']):
			print('Ran layer %d'%layer_id)
			ensure_path_exists('./results/' + db['data_name'] + '/' + str(layer_id))
			pth = './results/' + db['data_name'] + '/' + str(layer_id) + '/'

			zᶩ_vs_σᶩᐨᑊ = np.empty((db['n'], 0))
			ℍᶩ_目 = []
			ℍᶩᐨᑊ_目 = []
			σ_目 = []
			U_目 = []
			L_目 = []
			mostℍ_improved = 0
			missing_result = True

			for σᶩᐨᑊ in ȋ:
				[ṝ, zᶩ, ℍᶩ, ℍᶩᐨᑊ, σᶩ, ℱᴀ, U, L, ℍᶩ_best_σᶩ, rᶩᐩᑊ, Hⁿ] = self.find_σᶩᐨᑊ(rᶩᐨᑊ, σᶩᐨᑊ, layer_id)
				if ℍᶩ_best_σᶩ > last_best_ℍ and ℍᶩ_best_σᶩ > ℍᶩᐨᑊ and missing_result:
					missing_result = False
					last_best_ℍ = ℍᶩ_best_σᶩ
					mostℍ_improved = ℍᶩ_best_σᶩ - ℍᶩᐨᑊ
					best_ℍᶩ = ℍᶩ_best_σᶩ
					best_ℍᶩᐨᑊ = ℍᶩᐨᑊ
					best_ṝ = ṝ
					best_zᶩ = zᶩ
					best_ℱᴀ = ℱᴀ
					best_σᶩ = σᶩ
					best_σᶩᐨᑊ = σᶩᐨᑊ
					best_rᶩᐩᑊ = rᶩᐩᑊ
					best_Hⁿ = Hⁿ
	
				zᶩ_vs_σᶩᐨᑊ = np.hstack((zᶩ_vs_σᶩᐨᑊ, zᶩ))
				ℍᶩ_目.append(ℍᶩ)
				ℍᶩᐨᑊ_目.append(ℍᶩᐨᑊ)
				σ_目.append(σᶩᐨᑊ)
				U_目.append(U)
				L_目.append(L)
				int_info = [zᶩ_vs_σᶩᐨᑊ, ℍᶩ_目, ℍᶩᐨᑊ_目, σ_目, pth, U_目, L_目]
				#import pdb; pdb.set_trace()	
			
			self.record_layer_parameters([best_ṝ , best_zᶩ , best_ℍᶩ , best_ℍᶩᐨᑊ , best_σᶩ , best_σᶩᐨᑊ , best_ℱᴀ, pth])
			self.record_layer_internal_info(int_info)
			self.plot_internal_info(pth, repeat+1, layer_id, rᶩᐨᑊ, best_zᶩ, best_rᶩᐩᑊ)

			layer_id += 1
			rᶩᐨᑊ = best_zᶩ
			#ȋ = np.flip(np.linspace(0, best_σᶩ, 15))
			ȋ = np.flip(np.linspace(db['smallest_σ'], best_σᶩ, 15))


			print('\tσ : %.3f , ℍ* : %.3f'%(best_σᶩᐨᑊ, best_Hⁿ))
			if best_Hⁿ > 0.97: break;

