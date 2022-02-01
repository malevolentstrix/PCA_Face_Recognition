#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import argparse
import matplotlib.pyplot as plt
import numpy as np
import image
import eig
from sklearn import svm
import video

CAPTURED_VARIANCE = 0.9

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="Complete path with sub-directories containing images")
args = parser.parse_args()

print("Loading images...")
grayscaled_images, categories = image.load_images(args.directory)

# Normalizing
grayscaled_images = grayscaled_images / image.NORMALIZE_FACTOR

print("Subtracting mean face...")
# Subtract the mean face
mean_face =  np.mean(grayscaled_images, 0)
grayscaled_images -= mean_face

# Show the mean face
#fig, axes = plt.subplots(1,1)
#axes.imshow(np.reshape(mean_face,[image.VERTICAL_SIZE,image.HORIZONTAL_SIZE])*image.NORMALIZE_FACTOR,cmap='gray')
#fig.suptitle('Imagen media')
#plt.show()

# Let's call A the matrix with the images.
# We need the eigenfaces. These are the columns of the matrix V in the SVD decomposition of A.
# The columns of V are the eigenvectors of what we call the right covariance matrix (AᵀxA).
# The problem is that this matrix is too large (10304x10304). So instead we calculate the columns U in the SVD decomposition.
# The columns of U are the eigenvectors of the left covariance matrix (AxAᵀ). With the columns of U, we can get the columns of V by knowing that:
# Aᵀ x u = σ x v
# where u is a column of U, σ is the corresponding singular value and v is the corresponding column of V
left_covariance_matrix = grayscaled_images.dot(grayscaled_images.transpose())

print("Computing eigenvalues...")
eigen_values = eig.sorted_eigen_values(left_covariance_matrix)
total_eigen_values_sum = sum(eigen_values)
partial_eigen_value_sum = 0
used_eigen_faces = 0
for ev in eigen_values:
	partial_eigen_value_sum += ev
	used_eigen_faces += 1
	if (partial_eigen_value_sum/total_eigen_values_sum > CAPTURED_VARIANCE):
		break

first_left_singular_vector = eig.inverse_iteration(left_covariance_matrix, eigen_values[0])
eigen_face = grayscaled_images.transpose().dot(first_left_singular_vector)/np.sqrt(eigen_values[0])
# First eigenface according to us
#eigen1 = (np.reshape(eigen_face,[image.VERTICAL_SIZE, image.HORIZONTAL_SIZE]))*image.NORMALIZE_FACTOR
#fig, axes = plt.subplots(1,1)
#axes.imshow(eigen1,cmap='gray')
#fig.suptitle('First eigenface according to us')
#plt.show()

# First eigenface according to pfierens
#U,S,V = np.linalg.svd(grayscaled_images,full_matrices = False)
#eigen1 = (np.reshape(V[0,:],[image.VERTICAL_SIZE,image.HORIZONTAL_SIZE]))*image.NORMALIZE_FACTOR
#fig, axes = plt.subplots(1,1)
#axes.imshow(eigen1,cmap='gray')
#fig.suptitle('First eigenface according to pfierens')
#plt.show()

# Compute the first eigenfaces
eigen_faces = list()
print("Computing " + str(used_eigen_faces) + " eigenfaces...")
for i in range(used_eigen_faces):
	left_singular_vector = eig.inverse_iteration(left_covariance_matrix, eigen_values[i])
	eigen_face = grayscaled_images.transpose().dot(left_singular_vector)/np.sqrt(eigen_values[0])
	eigen_face = [item for sublist in eigen_face for item in sublist]	# The above computation yields the eigenface as a list of one-dimentional lists, so we flatten it
	eigen_faces.append(eigen_face)
eigen_faces = np.asarray(eigen_faces)

# Project
print("Projecting images...")
projected_images = np.dot(grayscaled_images, eigen_faces.transpose())

# Classify
print("Classifying...")
classifier = svm.LinearSVC()
image_classes = [category for category in categories for _ in range(image.IMAGES_PER_DIRECTORY)]
classifier.fit(projected_images, image_classes)

# Predict
video.recognize_faces(mean_face, eigen_faces, classifier)
