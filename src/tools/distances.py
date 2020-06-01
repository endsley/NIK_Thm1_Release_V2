
import numpy as np
import sklearn.metrics

#目 = list
#田 = matrix
#Ł = label
#ⵙ = center

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
		mse = mse + np.sum(Δϰ*Δϰ)

	return mse/n

def CrossEntropy(X,Y):		# X is assumed to be in IDS
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
		mse = mse + np.sum(Δϰ*Δϰ)

	return mse/n
