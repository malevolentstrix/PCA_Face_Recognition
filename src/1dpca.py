import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import timeit
from utils import get_faces, get_person_num, show_sample_faces, split_train_test,get_stats


faces = get_faces(zipfile_path="./Grp13Dataset.zip")
show_sample_faces(faces=faces)

faceshape = list(faces.values())[0].shape
print("Face image shape:", faceshape)

classes = set(filename.split("/")[0] for filename in faces.keys())
print("Number of classes:", len(classes))
print("Number of images:", len(faces))

facematrix = []
facelabel = []

training_set, testing_set = split_train_test(zipfilepath="./Grp13Dataset.zip")

start = timeit.default_timer()
for key, val in training_set.items():
    # fig 1
    facematrix.append(val.flatten())
    facelabel.append(key.split("/")[0])

facematrix = np.array(facematrix)

pca = PCA().fit(facematrix)

n_components = 40
eigenfaces = pca.components_[:n_components]


fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
for i in range(16):
    if(i<len(eigenfaces)):
        break
    axes[i % 4][i//4].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
print("Showing the eigenfaces")
plt.show()
# eqn 4
weights = eigenfaces @ (facematrix - pca.mean_).T
print("Shape of the weight matrix:", weights.shape)


def get_best_match(filename):
    query = faces[filename].reshape(1, -1)
    # eqn 22
    query_weight = eigenfaces @ (query - pca.mean_).T
    euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
    best_match = np.argmin(euclidean_distance)
    #print("Best match %s with Euclidean distance %f" %  (facelabel[best_match], euclidean_distance[best_match]))

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 6))
    person_num, img_num = get_person_num(filename=filename)

    axes[0].imshow(query.reshape(faceshape), cmap="gray")
    axes[0].set_title("Query - Person " + str(person_num))
    axes[1].imshow(facematrix[best_match].reshape(faceshape), cmap="gray")
    axes[1].set_title("Best match - Person " + str((best_match//8) + 1))
    #plt.show()
    # uncomment the above line to see individual results of testing faces
    return (((best_match//8) + 1), person_num)

stop = timeit.default_timer()

correct_pred = 0
wrong_pred = 0
total_pred = 0
for key, val in testing_set.items():       
    predicted, actual = get_best_match(filename=key)
    total_pred += 1
    if predicted == actual:
        correct_pred += 1
    else:
        wrong_pred += 1

Correct_Predic, Wrong_Predic, Accuracy = get_stats(correct_pred,wrong_pred,total_pred)
print(f"Correct prediction: ",Correct_Predic)
print(f"Wrong prediction: ",Wrong_Predic)

print(f"Accuracy: ",Accuracy,"%")
print(f"Time Taken: ",round(stop-start,3),"s")
