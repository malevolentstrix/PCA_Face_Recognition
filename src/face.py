import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

faces = {}
with zipfile.ZipFile("./Grp13Dataset.zip") as facezip:
    for filename in facezip.namelist():
        if not filename.endswith(".jpg"):
            continue
        with facezip.open(filename) as image:

            faces[filename] = cv2.imdecode(np.frombuffer(
                image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
faceimages = list(faces.values())[-16:]
for i in range(16):
    axes[i % 4][i//4].imshow(faceimages[i], cmap="gray")
print("Showing sample faces")
plt.show()

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
    facematrix.append(val.flatten())
    facelabel.append(key.split("/")[0])

facematrix = np.array(facematrix)

pca = PCA().fit(facematrix)

n_components = 50
eigenfaces = pca.components_[:n_components]


fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
for i in range(16):
    axes[i % 4][i//4].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
print("Showing the eigenfaces")
plt.show()


weights = eigenfaces @ (facematrix - pca.mean_).T
print("Shape of the weight matrix:", weights.shape)


query = faces["Grp13Person39/390_39.jpg"].reshape(1, -1)
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


query = faces["Grp13Person40/391_40.jpg"].reshape(1, -1)
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
