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
        a = im.imread(mypath + dire + '/{}'.format(k) + '.pgm')
        images[imno,:] = (np.reshape(a,[1,areasize])-127.5)/127.5
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
        a = im.imread(mypath + dire + '/{}'.format(k) + '.pgm')
        imagetst[imno,:]  = (np.reshape(a,[1,areasize])-127.5)/127.5
        persontst[imno,0] = per
        imno += 1
    per += 1

#KERNEL: polinomial de grado degree
degree = 2
K = (np.dot(images,images.T)/trnno+1)**degree
#K = (K + K.T)/2.0
        
#esta transformación es equivalente a centrar las imágenes originales...
unoM = np.ones([trnno,trnno])/trnno
K = K - np.dot(unoM,K) - np.dot(K,unoM) + np.dot(unoM,np.dot(K,unoM))


#Autovalores y autovectores
eigen_values = eig.sorted_eigen_values(K)
eigen_vectors = list()
for ev in eigen_values:
    eigen_vector = eig.inverse_iteration(K, ev)
    eigen_vectors.append(eigen_vector)

eigen_vectors = np.asarray(eigen_vectors).transpose()[0]

w = np.asarray(eigen_values)
alpha = eigen_vectors
lambdas = w/trnno
lambdas = w

for col in range(alpha.shape[1]):
    alpha[:,col] = alpha[:,col]/np.sqrt(lambdas[col])

#pre-proyección
improypre   = np.dot(K.T,alpha)
unoML       = np.ones([tstno,trnno])/trnno
Ktest       = (np.dot(imagetst,images.T)/trnno+1)**degree
Ktest       = Ktest - np.dot(unoML,K) - np.dot(Ktest,unoM) + np.dot(unoML,np.dot(K,unoM))
imtstproypre= np.dot(Ktest,alpha)

nmax = alpha.shape[1]
accs = np.zeros([nmax,1])
for neigen in range(1,nmax):
    #Me quedo sólo con las primeras autocaras   
    #proyecto
    improy      = improypre[:,0:neigen]
    imtstproy   = imtstproypre[:,0:neigen]
        
    #SVM
    #entreno
    clf = svm.LinearSVC()
    clf.fit(improy,person.ravel())
    accs[neigen] = clf.score(imtstproy,persontst.ravel())
    print('{0},{1}'.format(neigen,accs[neigen][0]))
    #print('Precisión con {0} autocaras: {1} %\n'.format(neigen,accs[neigen]*100))

fig, axes = plt.subplots(1,1)
axes.semilogy(range(nmax),(1-accs)*100)
axes.set_xlabel('No. autocaras')
axes.grid(which='Both')
fig.suptitle('Error')

