from Util import *
from math import *
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib
from numpy.linalg import pinv, det




if __name__ == "__main__" :
	Files = ListAllFiles( myPath = "/home/srinath/Desktop/SpeakerRecognition/User1/UserTrainData")
	threshold = 0.03
	lenFiles = len(Files)
	audiolist = []
	sample_rates = []
	for x in range(lenFiles):
		audio, sample_rate = ReadAudio( Files[x] )
		audio = audio/audio.max()
		left = ThresholdIndex( audio, threshold )
		right = ThresholdIndex ( np.flip(audio), threshold )
		audiolist.append(audio[left:-(right+1)])
		sample_rates.append(sample_rate)

	stack = librosa.feature.mfcc(y = audiolist[0], sr = sample_rates[x], n_mfcc = 26).transpose()[:, 1:13]
	for x in range(lenFiles-1):
		features = librosa.feature.mfcc(y = audiolist[x+1], sr = sample_rates[x+1], n_mfcc = 26)
		features = features.transpose()[:, 1:13]
		np.concatenate([stack, features])

	savenp(stack, "/home/srinath/Desktop/SpeakerRecognition/User1/UserFeatures/stack")

	ubm_path = "/home/srinath/Desktop/SpeakerRecognition/UBM_DATA/GMM_precise.joblib.pkl"
	gmm_ubm = joblib.load(ubm_path)
	ubm_means = gmm_ubm.means_
	ubm_covars = gmm_ubm.covariances_
	ubm_weights = gmm_ubm.weights_



	feature = loadnp("/home/srinath/Desktop/SpeakerRecognition/User1/UserFeatures/stack.npy")

	n_gaussians = len(ubm_weights)
	n_features = feature.shape[0]
	feature_len = feature.shape[1]


	Pr = np.zeros([n_gaussians,n_features])
	n = np.zeros([n_gaussians])
	E = np.zeros([n_gaussians, feature_len, 1])
	E_2 = np.zeros([n_gaussians, feature_len, feature_len])
	

	weights_cap = np.zeros([n_gaussians])
	means_cap = np.zeros([n_gaussians, feature_len, 1])
	covars_cap = np.zeros([n_gaussians, feature_len, feature_len] )
	
	#Relevance Factor
	r = 16

	alpha = np.zeros([n_gaussians])

	for i in range(n_gaussians):
		for t in range( n_features ) :
			Pr[i,t] = (ubm_weights[i]*Gaussian( mean = ubm_means[i], covar = ubm_covars[i], vector = feature[t] ))
	
	for t in range( n_features ):
		sm = sum(Pr[:,t])
		Pr[:,t] = Pr[:,t]/sm

	for i in range(n_gaussians):
		n[i] = sum(Pr[i,:])
		sm = np.zeros(feature_len)[np.newaxis].T
		sm_2 = np.zeros([feature_len, feature_len])
		for t in range( n_features ) :
			sm = sm + Pr[i, t]*feature[t][np.newaxis].T
			sm_2 = sm + Pr[i,t]*(np.matmul(feature[t].T, feature[t]))
		E[i] = (1/n[i])*sm
		E_2[i] = (1/n[i])*sm_2

	for i in range(n_gaussians):
		alpha[i] = n[i]/(n[i] + r )

	for i in range(n_gaussians) :
		weights_cap[i] = (alpha[i]*n[i]/n_features + (1- alpha[i] )*ubm_weights[i])
		means_cap[i] = (alpha[i]*E[i] + (1- alpha[i])*ubm_means[i][np.newaxis].T)
		covars_cap[i] = (alpha[i]*E_2[i] + (1- alpha[i])*(ubm_covars[i] + np.matmul(ubm_means[i][np.newaxis].T, ubm_means[i][np.newaxis])) - np.matmul(means_cap[i], means_cap[i].T))
		
	weights_sum = sum(weights_cap)
	weights_cap = weights_cap/weights_sum	


	#Saving Weights, Means, Covars
	savenp( arr = weights_cap, filename = "weights", Path = "/home/srinath/Desktop/SpeakerRecognition/User1/UserFeatures")
	savenp( arr = means_cap, filename = "means", Path = "/home/srinath/Desktop/SpeakerRecognition/User1/UserFeatures")
	savenp( arr = covars_cap, filename = "covars", Path = "/home/srinath/Desktop/SpeakerRecognition/User1/UserFeatures")
