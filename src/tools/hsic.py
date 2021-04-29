
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

def ℍ(X,Y, σ, Kᵪ_type='linear', Kᵧ_type='linear'):	# compute normalized HSIC between X,Y
	n = X.shape[0]

	if Kᵪ_type == 'linear': Kᵪ = X.dot(X.T)
	if Kᵧ_type == 'linear': Kᵧ = Y.dot(Y.T)
	if Kᵪ_type == 'Gaussian': 
		γ = 1.0/(2*σ*σ)
		Kᵪ = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ)

	#plt.imshow(Kᵪ, cmap='Blues_r', interpolation='nearest') #cmap options = viridis,Blues_r,hot
	#plt.colorbar()
	#plt.show()
	#import pdb; pdb.set_trace()

	#np.fill_diagonal(Kᵪ, 0)
	#np.fill_diagonal(Kᵧ, 0)
	HKᵪ = Kᵪ - np.mean(Kᵪ, axis=0)					# equivalent to		HKᵪ = H.dot(Kᵪ)
	HKᵧ = Kᵧ - np.mean(Kᵧ, axis=0)                  # equivalent to		HKᵧ = H.dot(Kᵧ)

	Hᵪᵧ= np.sum(HKᵪ.T*HKᵧ)

	Hᵪ = np.linalg.norm(HKᵪ)						# equivalent to 	np.sqrt(np.sum(KᵪH*KᵪH))
	Hᵧ = np.linalg.norm(HKᵧ) 						# equivalent to 	np.sqrt(np.sum(KᵧH*KᵧH))
	H = Hᵪᵧ/( Hᵪ * Hᵧ )

	return [H, Kᵪ]

def double_center(Ψ):
	HΨ = Ψ - np.mean(Ψ, axis=0)								# equivalent to Γ = Ⲏ.dot(Kᵧ).dot(Ⲏ)
	HΨH = (HΨ.T - np.mean(HΨ.T, axis=0)).T
	return HΨH
