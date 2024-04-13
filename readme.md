# Alzheimer’s Disease Classification using Support Vector Machine (SVM)

This project demonstrates an Alzehmier disease classification system using Support Vector Machines (SVM) along with various feature extraction techniques including Histogram of Oriented Gradients (HOG), VGG16 features, and a hybrid approach combining both.

## Project Structure

- `data_loader.py`: Contains the `load_dataset` function to load images and labels from a specified directory.
- `feature_extraction.py`: Implements the `preprocess_and_feature_extraction` function to extract features from images using HOG, VGG16, or a hybrid method.
- `classifier.py`: Includes the `train_svm_classifier` function to train an SVM classifier using the extracted features.
- `main.py`: The main script that ties everything together, processing command-line arguments, executing the feature extraction, training the classifier, and evaluating the results.

## Installation

To set up the project environment to run the scripts, follow these steps:

1. Install the required Python packages:

`pip install -r requirements.txt`

2. To run the main script, use the following command:

`python main.py --image_size 28 --technique hog`

3. Note: if you want to use VGG image size must be 224. P.S. for quick evaluation with 70% accuracy you can use image size of 28 and hog feature extractor and if you want higher accuracy, have patience for 13 hours so that grid search can find best parameters when the input image size is 128.
