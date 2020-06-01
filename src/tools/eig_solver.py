
import numpy as np
import sklearn.metrics

# ð		: dimension
# 田	: Matrix
#	assume that M is symmetric

def SPCA_by_percentage(M, Mʙ, num_of_classes, percentage=0.99):
	Λ,V = np.linalg.eigh(M)

	Λ = np.flip(Λ)
	V = np.flip(V, axis=1)		#[:,0:2]
	cs = np.cumsum(Λ)/np.sum(Λ)
	residual_ð = np.sum(cs < percentage) + 1
	if residual_ð < num_of_classes: residual_ð = num_of_classes
	#return V[:, 0:residual_ð]

	##-------------

	Λ,V = np.linalg.eigh(Mʙ)

	Λ = np.flip(Λ)
	V = np.flip(V, axis=1)		#[:,0:2]
	cs = np.cumsum(Λ)/np.sum(Λ)
	
	return V[:, 0:residual_ð]



def PCA_by_percentage(M, percentage=0.99):
	Λ,V = np.linalg.eigh(M)

	Λ = np.flip(Λ)
	V = np.flip(V, axis=1)		#[:,0:2]
	cs = np.cumsum(Λ)/np.sum(Λ)
	residual_ð = np.sum(cs < percentage) + 1
	return V[:, 0:residual_ð]


def PCA_by_num_of_classes(M, c):
	Λ,V = np.linalg.eigh(M)

	Λ = np.flip(Λ)
	V = np.flip(V, axis=1)		#[:,0:2]
	return V[:, 0:c]


def get_Null_space(田):
	Λ,V = np.linalg.eigh(田)
	cs = np.cumsum(Λ)/np.sum(Λ)
	ð_of_Null_space = np.sum(cs < 0.0001)
	null_space = V[:, 0:ð_of_Null_space+0]
	return null_space


def get_Principal_Components(田):	# 田 is assumed to be symmetric
	#田 = np.random.random((10,10))
	#田 = 田 + 田.T
	Λ,V = np.linalg.eigh(田)
	Λ = np.absolute(Λ)
	idx = Λ.argsort()[::-1]   
	Λ = Λ[idx]
	V = V[:,idx]

	cs = np.cumsum(Λ)/np.sum(Λ)
	residual_ð = np.sum(cs < 0.99) + 1 
	Ꮴ = V[:,0:residual_ð]
	
	return Ꮴ
