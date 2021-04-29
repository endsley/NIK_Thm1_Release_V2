
import numpy as np
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import RBFSampler


#目 = list
#田 = matrix
#Ł = label
#ⵙ = center


def minmax_inter_intra_cluster_pairwise_distances(X,Y):	#目 implies list
	Ł_目 = np.unique(Y)
	σ_目 = []
	σ2_目 = []
	for i in Ł_目:
		indices = np.where(Y == i)
		subX = X[indices, :][0]
		pairDistances = sklearn.metrics.pairwise.pairwise_distances(subX)
		σ = np.max(pairDistances)
		σ_目.append(σ)


		for j in Ł_目:
			if i != j:
				indices = np.where(Y == j)
				subX2 = X[indices, :][0]
				pairDistances2 = sklearn.metrics.pairwise.pairwise_distances(subX, subX2)
				σ = np.min(pairDistances2)
				σ2_目.append(σ)
			
				

	return [np.min(σ2_目) ,np.max(σ_目)]


def mean_intra_cluster_pairwise_distances(X,Y):	#目 implies list
	Ł_目 = np.unique(Y)
	σ_目 = []
	for i in Ł_目:
		indices = np.where(Y == i)
		subX = X[indices, :][0]
		pairDistances = sklearn.metrics.pairwise.pairwise_distances(subX)
		σ = np.mean(pairDistances)
		σ_目.append(σ)


	ij_đ_田 = sklearn.metrics.pairwise.pairwise_distances(subX)
	#print('median : %.3f'% np.median(ij_đ_田))
	#print('max : %.3f'% np.max(ij_đ_田))
	#print('σ_目 : ', σ_目)
	return np.max(σ_目)

def cluster_center(X,Y):
	ð = X.shape[1]
	Ł_目 = np.unique(Y)
	ⵙ_田 = np.empty((0,ð))

	for Ł in Ł_目:
		indices = np.where(Y == Ł)
		subX = X[indices, :][0]
		m = np.mean(subX, axis=0)
		ⵙ_田 = np.vstack((ⵙ_田, m))

	return [Ł_目, ⵙ_田]


def scatter_ratio(X,Y):
	msk = Y.dot(Y.T)
	inv_msk = np.absolute(msk - 1)
	pD = sklearn.metrics.pairwise.pairwise_distances(X)

	sR = np.sum(pD*msk)/np.sum(pD*inv_msk)

	return sR 



def get_CSR(X,Y, σ):	
	
	γ = 1.0/(2*σ*σ)
	ℱᴀ = RBFSampler(gamma=γ, n_components=10000, random_state=1)	# random_state=1 forces repeatable results	
	R = ℱᴀ.fit_transform(X)

	[mean_intra_cos_sim, σ_intra_cos_sim] = mean_intra_cluster_cosine_similarity(R,Y)
	[mean_inter_cos_sim, σ_inter_cos_sim] = mean_inter_cluster_cosine_similarity(R,Y)

	CSR = mean_inter_cos_sim/mean_intra_cos_sim			# cosine similarity ratio
	return CSR


def Get_Cross_Entropy(X,Y, Yₒ):
	clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, Y)
	ŷ = -clf.predict_log_proba(X)
	n = X.shape[0]

	CE = np.sum(Yₒ*ŷ)/n
	return CE




def mean_intra_cluster_cosine_similarity(X,Y):
	Ł_目 = np.unique(Y)
	σ_目 = []
	cos_sim_目 = None

	for i in Ł_目:
		indices = np.where(Y == i)
		ϰ = X[indices, :][0]
		ḵ = ϰ.dot(ϰ.T)

		if cos_sim_目 is None:
			cos_sim_目 = ḵ.flatten()
		else:
			cos_sim_目 = np.hstack((cos_sim_目, ḵ.flatten()))

	avg_cos_sim = np.mean(cos_sim_目)
	σ_cos_sim = np.std(cos_sim_目)
	return [avg_cos_sim, σ_cos_sim]



def mean_inter_cluster_cosine_similarity(X,Y):
	Ł_目 = np.unique(Y)
	σ_目 = []
	cos_sim_目 = None

	for i in Ł_目:
		for j in Ł_目:
			if i != j:
				ϰ = X[np.where(Y == i), :][0]
				Ꮍ = X[np.where(Y == j), :][0]

				ḵ = ϰ.dot(Ꮍ.T)

				if cos_sim_目 is None:
					cos_sim_目 = ḵ.flatten()
				else:
					cos_sim_目 = np.hstack((cos_sim_目, ḵ.flatten()))


	avg_cos_sim = np.mean(cos_sim_目)
	σ_cos_sim = np.std(cos_sim_目)
	return [avg_cos_sim, σ_cos_sim]

def MSE(X,Y):		# X is assumed to be in IDS
	Ł_目 = np.unique(Y)
	σ_目 = []
	cos_sim_目 = None
	mse = 0
	n = len(Y)

	for i in Ł_目:
		indices = np.where(Y == i)
		ϰ = X[indices, :][0]
		m = np.mean(ϰ, axis=0)

		Δϰ = ϰ - m
		mse = mse + np.sum(np.linalg.norm(Δϰ*Δϰ, axis=1))/n

	return mse

