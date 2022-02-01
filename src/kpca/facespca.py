# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:32:14 2017

@author: pfierens
"""
from os import listdir
from os.path import join, isdir
from scipy import ndimage as im
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import eig

mypath      = 'att_faces/'
onlydirs    = [f for f in listdir(mypath) if isdir(join(mypath, f))]

#image size
horsize     = 92
versize     = 112
areasize    = horsize*versize

#number of figures
personno    = 44
trnperper   = 9
tstperper   = 1
trnno       = personno*trnperper
tstno       = personno*tstperper

#TRAINING SET
images = np.zeros([trnno,areasize])
person = np.zeros([trnno,1])
imno = 0
per  = 0
for dire in onlydirs:
    for k in range(1,trnperper+1):
        a = im.imread(mypath + dire + '/{}'.format(k) + '.pgm')/255.0
        images[imno,:] = np.reshape(a,[1,areasize])
        person[imno,0] = per
        imno += 1
    per += 1

#TEST SET
imagetst  = np.zeros([tstno,areasize])
persontst = np.zeros([tstno,1])
imno = 0
per  = 0
for dire in onlydirs:
    for k in range(trnperper,10):
        a = im.imread(mypath + dire + '/{}'.format(k) + '.pgm')/255.0
        imagetst[imno,:]  = np.reshape(a,[1,areasize])
        persontst[imno,0] = per
        imno += 1
    per += 1

#CARA MEDIA
meanimage = np.mean(images,0)
#resto la media
images  = [images[k,:]-meanimage for k in range(images.shape[0])]
imagetst= [imagetst[k,:]-meanimage for k in range(imagetst.shape[0])]

#PCA
images = np.asarray(images)
left_covariance_matrix = images.dot(images.transpose())
eigen_values = eig.sorted_eigen_values(left_covariance_matrix)

eigen_faces = list()
for i in range(len(images)):
	left_singular_vector = eig.inverse_iteration(left_covariance_matrix, eigen_values[i])
	eigen_face = images.transpose().dot(left_singular_vector)/np.sqrt(eigen_values[0])
	eigen_face = [item for sublist in eigen_face for item in sublist]	# The above computation yields the eigenface as a list of one-dimentional lists, so we flatten it
	eigen_faces.append(eigen_face)
eigen_faces = np.asarray(eigen_faces)

V = eigen_faces

#Primera autocara...

nmax = V.shape[0]
accs = np.zeros([nmax,1])
for neigen in range(1,nmax):
    #Me quedo sólo con las primeras autocaras
    B = V[0:neigen,:]
    #proyecto
    improy      = np.dot(images,np.transpose(B))
    imtstproy   = np.dot(imagetst,np.transpose(B))
        
    #SVM
    #entreno
    clf = svm.LinearSVC()
    clf.fit(improy,person.ravel())
    accs[neigen] = clf.score(imtstproy,persontst.ravel())
    print('{0},{1}'.format(neigen,accs[neigen][0]))

fig, axes = plt.subplots(1,1)
axes.semilogy(range(nmax),(1-accs)*100)
axes.set_xlabel('No. autocaras')
axes.grid(which='Both')
fig.suptitle('Error')

