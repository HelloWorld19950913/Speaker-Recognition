
#_____________________________Import Files_____________________________#

from Util import *
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib

#_____________________________----X----_____________________________#


def userTrain (user_id, path = None, max_iter = 500, tol = 0.1, gmm_path = None, gmm = None, feature = None) :
	sr = 16000
	if gmm_path != None:
		gmm = joblib.load(gmm_path)
	# audio , sr = ReadAudio( path )
	# audio = audio/np.max(audio)
	# threshold = 0.03
	# left = ThresholdIndex( audio, threshold )
	# right = ThresholdIndex( np.flip(audio), threshold)
	# audio = audio[left:-(right+1)]
	# feature = librosa.feature.mfcc(y = audio, sr = sr, n_mfcc = 26).transpose()[:, 1:13]
	if type(features) == type(None):
		if path == None :
			raise ValueError("Either path or features must be given")
		else:
			feature = getFeatures( path )

	if gmm == None:
		gmm = GaussianMixture(n_components = 20, max_iter = max_iter, verbose = 2, warm_start = True, tol = tol)
	gmm.fit(feature)
	_ = joblib.dump(gmm, "/home/srinath/Desktop/SpeakerRecognition/User" + str(user_id) + "GMM.joblib.pkl", compress= 9)
	if gmm.converged_ == True:
		print("GMM converged")
	else : 
		print("GMM did not converge")
	return gmm

def ext_test( ext_path ):

	# gmm = userTrain(user_id = 1, path = "/home/srinath/Desktop/SpeakerRecognition/train.wav")
	gmm = joblib.load("/home/srinath/Desktop/SpeakerRecognition/User1.joblib.pkl")
	ubm = joblib.load("/home/srinath/Desktop/SpeakerRecognition/UBM_DATA/GMM_precise.joblib.pkl")

	score_gmm_test = gmm.predict(getFeatures( path = "/home/srinath/Desktop/SpeakerRecognition/test.wav" ))
	score_gmm_ext = gmm.predict(getFeatures( path = ext_path ))
	score_gmm_train_data = gmm.predict(getFeatures( path = "/home/srinath/Desktop/SpeakerRecognition/train.wav" ))
	
	score_ubm_test = ubm.predict(getFeatures( path = "/home/srinath/Desktop/SpeakerRecognition/test.wav" ))
	score_ubm_ext = ubm.predict(getFeatures( path = ext_path ))
	score_ubm_train_data = ubm.predict(getFeatures( path = "/home/srinath/Desktop/SpeakerRecognition/train.wav" ))
	
	# print("score_gmm_test")
	# print(score_gmm_test)
	# print("score_ubm_test")
	# print(score_ubm_test)

	# print("score_gmm_train_data")
	# print(score_gmm_train_data)
	# print("score ubm train data")
	# print(score_ubm_train_data)

	# print("score_gmm_ext")
	# print(score_gmm_ext)
	# print("score_ubm_ext")
	# print(score_ubm_ext)

	overall_score_test = []
	overall_score_train_data = []
	overall_score_ext  = []

	for x in range(len(score_ubm_test)):
		if score_ubm_test[x] != 0 and score_gmm_test[x] != 0:
			overall_score_test.append(score_gmm_test[x]/score_ubm_test[x])
	
	for x in range(len(score_ubm_train_data)):
		if score_ubm_train_data[x] != 0 and score_gmm_train_data[x] != 0:
			overall_score_train_data.append(score_gmm_train_data[x]/score_ubm_train_data[x])

	for x in range(len(score_ubm_ext)):
		if score_ubm_ext[x] != 0 and score_gmm_ext[x] != 0:
			overall_score_ext.append(score_gmm_ext[x]/score_ubm_ext[x])

	# print("overall_score_test")
	# print(overall_score_test)

	# print("overall_score_train_data")
	# print(overall_score_train_data)
	
	# print("overall_score_ext")
	# print(overall_score_ext)
	

	print("avg_test")
	print(np.mean(overall_score_test))
	
	print("avg_test2")
	print(np.mean(overall_score_train_data))
	
	print("avg_ext")
	print(np.mean(overall_score_ext))
	print()
	print("---------------x---------------")
	print()

	# For Testing Overall Performance on many files
	# if np.mean(overall_score_ext) > 0.3:
	# 	return 1
	# return 0

	return 

if __name__ == "__main__" :

	Files = ListAllFiles( myPath = "/home/srinath/Desktop/SpeakerRecognition/User1/")
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

	savenp(stack, "/home/srinath/Desktop/SpeakerRecognition/User1/stack")
	# userTrain(user_id = 1, feature = stack, gmm_path = "/home/srinath/Desktop/SpeakerRecognition/User1.joblib.pkl")
	Fileslist = ListAllFiles ( myPath = "/home/srinath/Desktop/Timit_data")
	# count  = 0
	for x in range(10) :
		ext_test( ext_path = Fileslist[x] )
		# count = count + ext_test( ext_path = Fileslist[x] )
	# print(count)
	print("For Yes.wav")
	ext_test("/home/srinath/Desktop/SpeakerRecognition/ext.wav")