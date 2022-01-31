from itertools import count
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils import get_faces, show_sample_faces


def find_mean_mat(facematrix):
    #eqn 5
    #print(len(facematrix), len(facematrix[0]), len(facematrix[0][0]))
    sum_matr = np.empty((80,70), dtype=np.int8)
    for img in facematrix:
        sum_matr = np.add(img, sum_matr)
    #print (sum_matr/len(facematrix))
    return sum_matr/len(facematrix)

faces = get_faces(zipfile_path="./Grp13Dataset.zip")
#show_sample_faces(faces=faces)

faceshape = list(faces.values())[0].shape
print("Face image shape:", faceshape)

classes = set(filename.split("/")[0] for filename in faces.keys())
print("Number of classes:", len(classes))
print("Number of images:", len(faces))

facematrix = []
facelabel = []
for key, val in faces.items():
    if key.startswith("Grp13Person40/"):
        continue
    if key == "Grp13Person39/390_39.jpg":
        continue
    
    #facematrix[i] is i-th image
    facematrix.append(val)
    facelabel.append(key.split("/")[0])

facematrix = np.array(facematrix)

M = find_mean_mat(facematrix)

A=[]
for img in facematrix:
    #eqn 6
    A.append(img-M)
A= np.array(A)
print(A.shape)

C=[]
for i in A:
    #print(i.shape)
    C.append(i.transpose() @ i)

C = np.array(C)
C = np.divide(C, len(facematrix))
#eqn 7
print(C)

eigen_values, eigen_vectors = np.linalg.eig(C)


n_components = 50
eigenfaces = eigen_vectors[:n_components]


fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
for i in range(16):
    axes[i % 4][i//4].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
print("Showing the eigenfaces")
plt.show()

#eqn 4
weights = eigenfaces @ (facematrix - pca.mean_).T
print("Shape of the weight matrix:", weights.shape)


def get_best_match(filename):
    query = faces[filename].reshape(1, -1)
    #eqn 22
    query_weight = eigenfaces @ (query - pca.mean_).T
    euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
    best_match = np.argmin(euclidean_distance)
    print("Best match %s with Euclidean distance %f" %
          (facelabel[best_match], euclidean_distance[best_match]))

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 6))
    axes[0].imshow(query.reshape(faceshape), cmap="gray")
    axes[0].set_title("Query")
    axes[1].imshow(facematrix[best_match].reshape(faceshape), cmap="gray")
    axes[1].set_title("Best match")
    plt.show()


get_best_match(filename="Grp13Person39/390_39.jpg")
get_best_match(filename="Grp13Person40/391_40.jpg")
