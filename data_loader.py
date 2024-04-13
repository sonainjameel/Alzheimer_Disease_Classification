import os
from skimage.io import imread

def load_dataset(base_path):
    images = []
    labels = []
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                image = imread(file_path)
                images.append(image)
                labels.append(folder_name)
    return images, labels
