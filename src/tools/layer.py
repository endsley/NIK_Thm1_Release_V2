
import numpy as np
import sklearn.metrics

def multiply_by_Ws(rᶩ, Ws):
	z = np.empty((rᶩ.shape[0], 0))
	for ws in Ws:
		element_projection = rᶩ.dot(ws)
		element_projection = np.clip(element_projection, 0, 2)
		new_column = np.sum(element_projection, axis=1, keepdims=True)
		z = np.hstack((z,new_column))

	return z

def multiply_by_Ws_kernel_trick(z_in, Ws, σ):
	γ = 1/(2*σ*σ)
	n = z_in.shape[0]
	z = np.empty((0, n))

	for w in Ws:
		[ř, zi] = w
		try:
			Ƙ = sklearn.metrics.pairwise.rbf_kernel(zi, z_in, gamma=γ)
		except:
			import pdb; pdb.set_trace()

		row = np.atleast_2d(np.sum(Ƙ, axis=0)/ř)
		z = np.vstack((z,row))
	z = z.T			
	return z


class layer():
	def __init__(self, layer_info):
		[self.ℱᴀ, self.Wsₐ, self.zᶩ, self.σₐ, self.σᵦ, self.H, self.Hᶩᐨᑊ, self.use_Kernel_Trick] = layer_info

	def run_layer(self, z):
		if self.use_Kernel_Trick:
			zout = multiply_by_Ws_kernel_trick(z, self.Wsₐ, self.σₐ)
			return zout
		else:
			rᶩᐨᑊ = self.ℱᴀ.fit_transform(z)
			zout = multiply_by_Ws(rᶩᐨᑊ, self.Wsₐ)
			return zout


		
class layer_manager():
	def __init__(self, db):
		self.ℓ目 = []
		self.db = db

	def get_σ_list(self):
		σ目 = []

		for ℓ in self.ℓ目:
			σ目.append(ℓ.σₐ)

		return σ目

	def get_last_H(self):
		ℓ = self.ℓ目[-1]
		return ℓ.H

	def add_layer(self, ℓ):
		self.ℓ目.append(ℓ)
		self.σᵦ = ℓ.σᵦ


	def run_all_layers(self, X):
		Ȋ = X
		for ℓ in self.ℓ目:
			Ȋ = ℓ.run_layer(Ȋ)

		return Ȋ
