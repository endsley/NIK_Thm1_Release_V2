#!/usr/bin/env python

import sklearn.metrics
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from numpy import genfromtxt
import pyrfm.random_feature
import matplotlib.pyplot as plt
import pickle
import math
import os


from src.tools.path_tools import *
from src.tools.opt_gaussian import *
from src.tools.hsic import *
from src.tools.plot_lib import *
from src.tools.merge_images import *
from src.tools.file_write import *
from src.tools.layer import *
from src.tools.class_data_stats import *

class kchain():
	def __init__(self, db):	
		self.db = db
		db['Ky'] = db['Yₒ'].dot(db['Yₒ'].T)
		self.Γ = double_center(db['Ky'])
		self.ΣΓᵴ = np.sum(np.ceil(self.Γ)*self.Γ)
		self.class_mask = {}
		self.Γ_classes = {}
		
		self.inner_σ_range = self.get_σ_range(db, 1)

		for class_id in db['Ł_目']:
			Yₒ = np.copy(db['Yₒ'])
			indices = np.where(db['Y'] != class_id)[0]
			Yₒ[indices, :] = 0
			self.class_mask[class_id] = Yₒ.dot(Yₒ.T)
			self.Γ_classes[class_id] = self.class_mask[class_id]*self.Γ

		self.inv_mask = np.absolute(db['Ky'] - 1)
		self.betwee_class_Γ = self.Γ*self.inv_mask



	def obtain_avg_direction_kernel_trick_with_extra_dimensions(self, zᶩᐨᑊ, class_id, σ):
		db = self.db
		[rᶩᐨᑊ, ℱᴀ] = self.map_to_RKHS_via_RFF(σ, zᶩᐨᑊ)

		Ws = []
		for cid in class_id:
			indices = np.where(db['Y'] == cid)[0]
			r_class_i = rᶩᐨᑊ[indices, :].T

			# normalizing
			r_class_i_μ = np.sum(r_class_i, axis=1)
			rn = np.linalg.norm(r_class_i_μ) # use RFF to approximate the normalizer, but use kernel trick

			r_class_i = zᶩᐨᑊ[indices, :]	
			Ws.append( [rn, r_class_i] )

		#---------------------------
		if db['X'].shape[1] != zᶩᐨᑊ.shape[1]:
			Ŷ = np.argmax(zᶩᐨᑊ, axis=1)
			zs = zᶩᐨᑊ[db['Y'] != Ŷ]
			Ys = db['Y'][db['Y'] != Ŷ]
			for cid in class_id:
				indices = np.where(Ys == cid)[0]
				if indices.size != 0:
					r_class_i = zs[indices, :]
					Ws.append( [1, r_class_i] )
		return Ws


	def obtain_avg_direction_kernel_trick(self, zᶩᐨᑊ, class_id, σ):
		db = self.db
		[rᶩᐨᑊ, ℱᴀ] = self.map_to_RKHS_via_RFF(σ, zᶩᐨᑊ)

		Ws = []
		for cid in class_id:
			indices = np.where(db['Y'] == cid)[0]
			r_class_i = rᶩᐨᑊ[indices, :].T

			# normalizing
			r_class_i_μ = np.sum(r_class_i, axis=1)
			rn = np.linalg.norm(r_class_i_μ) # use RFF to approximate the normalizer, but use kernel trick

			r_class_i = zᶩᐨᑊ[indices, :]	
			Ws.append( [rn, r_class_i] )

		return Ws

	def obtain_avg_direction(self, rᶩ, class_id):
		db = self.db

		Ws = []				# Didn't use the mean embedding directly cus RFF's noise creates negative values
		for cid in class_id:
			indices = np.where(db['Y'] == cid)[0]
			r_class_i = rᶩ[indices, :].T

			# normalizing
			r_class_i_μ = np.sum(r_class_i, axis=0)
			rn = np.linalg.norm(r_class_i_μ)
			r_class_i = r_class_i/rn

			Ws.append(r_class_i)

		return Ws



	def get_σ_range(self, db, largest_σ):
		start_power = np.log(largest_σ)/np.log(10)
		σ_end = np.log10(db['smallest_σ'])
		ȋ = np.logspace(start_power, σ_end , db['number_of_σ_to_test'])
		
		return ȋ


	def map_to_RKHS_via_RFF(self, σ, z):
		db = self.db
		γ = 1.0/(2*σ*σ)

		ℱᴀ = RBFSampler(gamma=γ, n_components=db['RFF_ð'], random_state=1)	# random_state=1 forces repeatable results	
		Φz = ℱᴀ.fit_transform(z)
		return [Φz, ℱᴀ]

	def get_optimal_layer(self, zᶩᐨᑊ, σₐ, Hᶩᐨᑊ):     # zᶩᐨᑊ ->  ℱᴀᑊ, (rᶩ), Wsᵦ, ℱᴀᒾ -> rᶩᐩᑊ
		db = self.db

		if db['use_kernel_trick']:
			#	Pass through 1 layer
			#Wsₐ = self.obtain_avg_direction_kernel_trick(zᶩᐨᑊ, db['Ł_目'], σₐ)
			Wsₐ = self.obtain_avg_direction_kernel_trick_with_extra_dimensions(zᶩᐨᑊ, db['Ł_目'], σₐ)
			zᶩ = multiply_by_Ws_kernel_trick(zᶩᐨᑊ, Wsₐ, σₐ)
			ℱᴀ = None
		
			# Pass through 2nd layer
			σᵦ = get_opt_σ(zᶩ, db['Y'], Y_kernel='linear', min_σ=0.01)
			[Hᶩ,K] = ℍ(zᶩ, db['Yₒ'], σᵦ, Kᵪ_type='Gaussian', Kᵧ_type='linear')
		else:
			#	Pass through 1 layer
			[rᶩᐨᑊ, ℱᴀ] = self.map_to_RKHS_via_RFF(σₐ, zᶩᐨᑊ)
			Wsₐ = self.obtain_avg_direction(rᶩᐨᑊ, db['Ł_目'])
			zᶩ = multiply_by_Ws(rᶩᐨᑊ, Wsₐ)
		
			# Pass through 2nd layer
			σᵦ = get_opt_σ(zᶩ, db['Y'], Y_kernel='linear', min_σ=0.01)
			[Hᶩ,K] = ℍ(zᶩ, db['Yₒ'], σᵦ, Kᵪ_type='Gaussian', Kᵧ_type='linear')

		return layer([ℱᴀ, Wsₐ, zᶩ, σₐ, σᵦ, Hᶩ, Hᶩᐨᑊ, db['use_kernel_trick']])



	def exit_loop(self, σₐ, σ_range, ℍᶩᐨᑊ):
		db = self.db

		if σₐ == σ_range[-1]: 
			#print('exit1 ran out σ')
			return True
		if ℍᶩᐨᑊ > db['HSIC_exit_threshold']: 
			#print('exit2 hit hisc')
			return True

		return False

	def predict(self, X):
		ӯ = self.LM.run_all_layers(X)


	def get_network_output(self, X):
		return self.LM.run_all_layers(X)


	def save_layer_info(self, layer_id, pth, ℓ):
		layer_txt = 'layer: %d ,  σₐ: %.5f, σᵦ: %.5f, ℍᶩᐨᑊ: %.3f, ℍᶩ: %.3f'%(layer_id, ℓ.σₐ, ℓ.σᵦ , ℓ.ℍᶩᐨᑊ, ℓ.H)
		save_result(pth + 'layer_output_summary.txt', layer_txt)

	def fit(self, tenFold_id=0):
		db = self.db
		self.LM = layer_manager(db)

		#	 Initialization
		zᶩᐨᑊ = db['X']
		σᶩᐨᑊ = get_opt_σ(db['X'],db['Y'], Y_kernel='linear', min_σ=0.1)
		#[ℍᶩᐨᑊ,K] = ℍ(zᶩᐨᑊ, db['Yₒ'], σᶩᐨᑊ, Kᵪ_type='Gaussian', Kᵧ_type='linear')
		ℍᶩᐨᑊ = 0.1			# slow down the improvement rather than just jumping to optimum

		σ_range = self.get_σ_range(db, σᶩᐨᑊ)
		pth = './results/' + db['data_name'] + '/'
		ensure_path_exists(pth)

		for layer_id in np.arange(1, db['max_layer']):

			for σₐ in σ_range:				
				ℓ = self.get_optimal_layer(zᶩᐨᑊ, σₐ, ℍᶩᐨᑊ)
				#print(σₐ, ℍᶩᐨᑊ, ℓ.H)
				if ℍᶩᐨᑊ < ℓ.H:
					self.LM.add_layer(ℓ)
					
					self.save_layer_info(layer_id, pth, ℓ)
					zᶩᐨᑊ = ℓ.zᶩ						# the output of last layer zᶩ becomes the new input zᶩᐨᑊ
					ℍᶩᐨᑊ = ℓ.H
					#print('%.7f, %.3f, %.3f'%(σₐ, ℍᶩᐨᑊ, ℓ.H))
					#σnew = get_opt_σ(zᶩᐨᑊ,db['Y'], Y_kernel='linear', min_σ=0.01)
					#σ_range = self.get_σ_range(db, σnew)
					σ_range = self.get_σ_range(db, ℓ.σᵦ)
					break

			if self.exit_loop(σₐ, σ_range, ℍᶩᐨᑊ): break
