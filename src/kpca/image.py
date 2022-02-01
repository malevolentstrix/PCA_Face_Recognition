import numpy as np
from PIL import Image
from os import path, listdir

VERTICAL_SIZE = 112
HORIZONTAL_SIZE = 92
NORMALIZE_FACTOR = 255.0

IMAGES_PER_DIRECTORY = 10

def load_images(directory):
    """Load images and return them as a byte array representing the grayscale values."""
    subdirectories = [f for f in listdir(directory)]
    grayscale_images = [[],[]]
    categories = list()

    for face in subdirectories:
        images_paths = [(directory + "/" +  face + "/" +  f) for f in 
                listdir(directory + "/" + face)]
        for image_path in images_paths:
            grayscale_image = to_grayscale(image_path)
            grayscale_images[0].append(grayscale_image)
            grayscale_images[1].append(grayscale_image)
    return np.asarray(grayscale_images[0]), subdirectories

def to_grayscale(image_path):
    return np.asarray(list(Image.open(image_path).convert('L').tobytes()))
