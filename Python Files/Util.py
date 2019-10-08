
#_____________________________Import Modules_____________________________#

import os
from os.path import isdir, isfile, join
import numpy as np
import soundfile as sf
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.externals import joblib
from math import *
from numpy.linalg import pinv, det
import sys
import librosa

#_____________________________----X----_____________________________#




#_____________________________Utility Functions_____________________________#


#---------------------------------normalize---------------------------------#

def normalize ( arr ) :
	mx = np.max(abs(arr))
	if mx == 1:
		return arr
	return arr/mx

#---------------------------------savenp---------------------------------#

def savenp ( arr , filename, Path = "" ):
	if type(arr).__module__ is not np.__name__:
		#print("Array provided to save not of the type numpy, trying to convert into numpy array")
		arr = np.array(arr)
		#print("Successfully converted!!")

	np.save( join(Path, filename) , arr)
	print("np array provided is saved as " + filename)

#---------------------------------loadnp---------------------------------#

def loadnp ( filename, Path = "" ):
	if not filename.endswith(".npy"):
		filename = filename + ".npy"
	return np.load(join(Path , filename))

#---------------------------------ListAllFiles---------------------------------#

def ListAllFiles ( myPath = "/media/srinath/AC64112A6410F8B2/Timit_data", Extention = "wav" ):



	filePaths = []

	for root, dirs, files in os.walk(myPath) :
		for file in files:
			if file.endswith(Extention):
				filePaths.append(join(root, file))

	return filePaths

#---------------------------------FitGMM---------------------------------#

def FitGMM ( n_components, features, covariance_type = "full", tol = .001, verbose = 2 ,max_iter=100, n_init=1, warm_start=False, verbose_interval = 10):

	if type(features).__module__ is not np.__name__ :
		raise TypeError("features should be of type numpy")

	gmm = GaussianMixture( n_components = n_components, covariance_type = covariance_type, tol = tol, verbose = verbose, n_init = n_init, warm_start = warm_start, verbose_interval = verbose_interval  )
	gmm.fit( features )

	if gmm.converged_ == True :
		return gmm.weights_ , gmm.means_, gmm.covariances_, n_iter_
	else :
		print("GMM did not converge, either increase max_iter or set warm_start = True")
		sys.exit()		

#---------------------------------ReadAudio---------------------------------#

def ReadAudio ( filename, Path = "") :

	filepath = join(Path, filename)
	audio, sample_rate = sf.read( filepath )
	return audio, sample_rate

#---------------------------------ThresholdIndex---------------------------------#

def ThresholdIndex ( audio, threshold) :
  for x in range(len(audio)):
    if audio[x]>threshold :
      return x
#---------------------------------getFeatures---------------------------------#

def getFeatures ( path ):
	audio , sr = ReadAudio( path )
	audio = audio/np.max(audio)
	if audio.ndim == 2:
		y = audio
		audio = np.zeros(len(y))
		for x in range(len(audio)):
			audio[x] = np.mean(y[x])
			
	threshold = 0.03
	left = ThresholdIndex( audio, threshold )
	right = ThresholdIndex( np.flip(audio), threshold)
	audio = audio[left:-(right+1)]
	feature = librosa.feature.mfcc(y = audio, sr = sr, n_mfcc = 26).transpose()[:, 1:13]
	return feature

#---------------------------------Gaussian---------------------------------#

def Gaussian (mean, covar, vector) :
	p = len(mean)
	if mean.ndim == 1:
		mean = mean[np.newaxis]
		mean = mean.T
	if vector.ndim == 1:
		vector = vector[np.newaxis]
		vector = vector.T

	return (exp((-1/2)*np.matmul(np.matmul((vector-mean).T,pinv(covar)),(vector-mean)))/sqrt(((2*pi)**p)*abs(det(covar))))

#_____________________________End of Util_____________________________#
