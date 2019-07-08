import os
from os.path import isdir, isfile, join
import librosa
import numpy as np
import soundfile as sf
from sklearn.mixture import GaussianMixture
import sys

#---------------------------------RemoveNoise---------------------------------#
# Needs Work
def RemoveNoise ( y, scale = 20 ) :

	mean = np.mean(y)

	for x in range(len(y)):
		if y[x] < mean/scale:
			y[x] = 0

#---------------------------------normalize---------------------------------#

def normalize ( arr ) :
	mx = np.max(abs(arr))
	if mx == 1:
		return arr
	return arr/mx

#---------------------------------savenp---------------------------------#

def savenp ( arr , filename, Path = "" ):
	if type(arr).__module__ is not np.__name__:
		print("Array provided to save not of the type numpy, trying to convert into numpy array")
		arr = np.array(arr)
		print("Successfully converted!!")

	np.save( join(Path, filename) , arr)
	print("np array provided is saved as " + filename)

#---------------------------------loadnp---------------------------------#

def loadnp ( filename, Path = "" ):
	if not filename.endswith(".npy"):
		filename = filename + ".npy"
	return np.load(join(Path , filename))

#---------------------------------Features---------------------------------#
# Needs work
def Features( filePaths frame_length = 25 ):

	features = []

	for filePath in filePaths:
		y, sr = librosa.load(filePath)
		y = RemoveNoise(y)

		n_mfcc = frame_length*sr/1000
		features.append(librosa.feature.mfcc(y = y, sr = sr, n_mfcc = n_mfcc))

	return features

#---------------------------------ListAllFiles---------------------------------#

def ListAllFiles ( myPath = "/media/srinath/AC64112A6410F8B2/Timit_data", Extention = "wav" ):

	if Extention[0] is not "."
		Extention = "." + Extention

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

def ReadAudio ( filename, Path = " ") :

	filepath = join(Path, filename)
	audio, sample_rate = sf.read("filepath")
	return audio, sample_rate

def GetUser( id = None ) :

	if id is None :

		loadnp(  )


