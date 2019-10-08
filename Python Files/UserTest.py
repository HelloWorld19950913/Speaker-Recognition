from Util import *
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib
from math import *


def score( features, ubm_weights, ubm_means, ubm_covars, usr_weights, usr_means, usr_covars, C = 5 ):
	n_features = len(features)
	n_gaussians = len(ubm_weights)

	score = np.zeros([n_features, n_gaussians])
	usr_score = np.zeros([n_features, C])

	idx = np.zeros([n_features, C], dtype = "int64")

	for i in range(n_features):
		for n in range(n_gaussians):
			score[i, n] = ubm_weights[n]*Gaussian (mean = ubm_means[n], covar = ubm_covars[n], vector = features[i][np.newaxis].T)
		idx[i] = (-score[i]).argsort()[:5]

	for i in range(n_features):
		k = 0
		for n in idx[i]:
			usr_score[i, k] = (usr_weights[n])*Gaussian (mean = usr_means[n], covar = usr_covars[n], vector = features[i][np.newaxis].T)
			k = k + 1

	log_likelyhood = 0
	for i in range(n_features):
		log_likelyhood =  log_likelyhood + log(sum(usr_score[i])) - log(sum(score[i, idx[i]]))

	#Average log_likelyhood
	return log_likelyhood/n_features



def main():
	UserTestPath = "/home/srinath/Desktop/SpeakerRecognition/User1/UserTestData"
	UserFeaturePath = "/home/srinath/Desktop/SpeakerRecognition/User1/UserFeatures/"
	usr_weights = loadnp(UserFeaturePath + "weights.npy")
	usr_means = loadnp(UserFeaturePath + "means.npy")
	usr_covars = loadnp(UserFeaturePath + "covars.npy")

	ubm_path = "/home/srinath/Desktop/SpeakerRecognition/UBM_DATA/GMM_precise.joblib.pkl"
	gmm_ubm = joblib.load(ubm_path)
	ubm_means = gmm_ubm.means_
	ubm_covars = gmm_ubm.covariances_
	ubm_weights = gmm_ubm.weights_

	UserTestFiles = ListAllFiles(UserTestPath)

	test_features = []
	for test_file in UserTestFiles:
		test_features.append(getFeatures(test_file))
	

	#Top C scoring mixtures
	C = 5
	scores = []
	for test_file in test_features:		
		scores.append(score( test_features3, ubm_weights, ubm_means, ubm_covars, usr_weights, usr_means, usr_covars, C ))

	print("Format")
	print("---------------------")
	print("score")
	print("exp(score)")
	print("---------------------")
	print()
	
	for score in scores:
		print("---------------------")
		print(score)
		print(exp(score))
		print("---------------------")

if __name__ == '__main__':
	main()