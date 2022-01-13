import os
import time
import dlib
import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


os.chdir('/home/hhmt/projects/tma/face_recog/')
train_size = 6407
val_size = 2742
test_size = 0
face_size = 224
project_dimension = 3
alignment_size = face_size * project_dimension
k_neighbors = 1


with open("train_data.pkl", "rb") as f:
    x_train = pickle.load(f)
x_train = x_train.reshape(-1, face_size, face_size, 3)
x_train = x_train[:,:,:,0]

with open("train_labels.pkl", "rb") as f:
    y_train = pickle.load(f)


with open("validation_data.pkl", "rb") as f:
    x_val = pickle.load(f)
x_val = x_val.reshape(-1, face_size, face_size, 3)
x_val = x_val[:,:,:,0]

with open("validation_labels.pkl", "rb") as f:
    y_val = pickle.load(f)


# Calculate mean of all train data
mean = np.zeros((face_size, face_size))
for i in range(0, face_size):
    for j in range(0, face_size):
        for t in range(0, train_size):
            mean[i][j] += x_train[t][i][j]
        mean[i][j] /= train_size


# Generate Scatter Matrix
scatter_matrix = np.zeros((face_size, face_size))
for i in range(0, train_size):
    scatter_matrix += (x_train[i,:,:] - mean).dot((x_train[i,:,:] - mean).T)


# Generate projection vector
projection_vector = np.zeros((face_size, 0))

eigen_values, eigen_vectors = np.linalg.eig(scatter_matrix)
eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(0, face_size)]
eigen_pairs.sort(key=lambda x: x[0], reverse=True)

for i in range(0, project_dimension):
    projection_vector = np.hstack((projection_vector, eigen_pairs[i][1].reshape(face_size, 1)))


# Transform x_train 
tmp = []
for i in range(0, train_size):
    tmp.append(projection_vector.T.dot(x_train[i]))
x_train = np.zeros((train_size, project_dimension, face_size))
x_train = np.concatenate([tmp])
    

# Align x_train to 1D
x_train = x_train.reshape(train_size, alignment_size)


# Transform x_val
tmp = []
for i in range(0, val_size):
    tmp.append(projection_vector.T.dot(x_val[i]))
x_val = np.zeros((val_size, project_dimension, face_size))
x_val = np.concatenate([tmp])


# Align x_val to 1D
x_val = x_val.reshape(val_size, alignment_size)


# 
scaler = StandardScaler()  
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)

classifier = KNeighborsClassifier(n_neighbors=k_neighbors)
classifier.fit(x_train, y_train)

y_pred = np.ravel(classifier.predict(x_val))

print(classification_report(y_val, y_pred))  
print(confusion_matrix(y_val, y_pred))
print(accuracy_score(y_val, y_pred))

