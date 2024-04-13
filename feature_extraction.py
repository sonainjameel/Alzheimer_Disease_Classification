from skimage.transform import resize
from skimage.feature import hog
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
import numpy as np
from skimage.color import gray2rgb, rgb2gray

# Load VGG16 model, with weights pre-trained on ImageNet
base_model = VGG16(weights='imagenet')
# Create a new model, using the output of an intermediate layer in VGG16
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def preprocess_and_feature_extraction(images, technique='hog', image_size=(224, 224)):

    features = []
    for image in tqdm(images, desc=f'Extracting {technique.upper()} Features'):
        # Resize the image first to ensure it's the correct dimensions for feature extraction
        resized_img = resize(image, image_size, anti_aliasing=True)

        if technique == 'hog':
            # HOG feature extraction
            feat = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True)
        elif technique == 'vgg' or technique == 'hybrid':
            # Check if the image is grayscale (only one color channel)
            if resized_img.ndim == 2 or resized_img.shape[2] == 1:
                resized_img = gray2rgb(resized_img)  # Convert grayscale to RGB
            # Convert the image to array format for VGG16 processing
            img_array = img_to_array(resized_img)
            expanded_img = np.expand_dims(img_array, axis=0)  # Expand dims to fit model input
            preprocessed_img = preprocess_input(expanded_img)  # Preprocess the image
            vgg_feat = model.predict(preprocessed_img)[0]  # Extract features

            if technique == 'vgg':
                feat = vgg_feat
            elif technique == 'hybrid':
                # Continue with HOG feature extraction for hybrid
                resized_img = rgb2gray(resized_img)
                hog_feat = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True)
                feat = np.concatenate((vgg_feat, hog_feat))
        else:
            raise ValueError("Technique not supported. Choose 'hog', 'vgg', or 'hybrid'.")

        features.append(feat)
    return features
