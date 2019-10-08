from Util import *
import soundfile as sf
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
import numpy as np

'''
****************
For Testing we read the features extracted
for testing from Features2 folder instead of 
Features folder and save the featurelist in 
the name of featurelist2 instead of featurelist
****************
'''

'''
****************
Changing the name to Features2
****************
'''
FeatureFiles = ListAllFiles("/home/srinath/Desktop/SpeakerRecognition/Features2", Extention = "npy")
# print(len(FeatureFiles))
stack = loadnp(FeatureFiles[0])[1:13, :].transpose()
print("done0")
# print(stack.shape)
FeatureFiles.remove(FeatureFiles[0])
lenFeatureFiles = len(FeatureFiles)
for x in range(lenFeatureFiles) :
  feature = loadnp(FeatureFiles[x]).transpose()
  stack = np.concatenate((stack, feature[:, 1:13]),)
  # print(stack.shape)
  print("done"+ str(x+1))
print(stack.shape)


'''
****************
Saving the feature stack in the name
of featurelist2
****************
'''
savenp( arr = stack, filename = "/home/srinath/Desktop/SpeakerRecognition/FeatureStack/featurelist2")
