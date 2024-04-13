from data_loader import load_dataset
from feature_extraction import preprocess_and_feature_extraction
from classifier import train_svm_classifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--image_size', type=int, default=28, help='Image size for resizing (e.g., 128 for 128x128)')
    parser.add_argument('--technique', type=str, default='hog', choices=['hog', 'vgg', 'hybrid'], help='Feature extraction technique')

    args = parser.parse_args()

    folder = "./Dataset"
    image_size = (args.image_size, args.image_size)
    images, labels = load_dataset(folder)
    features = preprocess_and_feature_extraction(images, technique=args.technique, image_size=image_size)
    y_test, y_pred, clf = train_svm_classifier(features, labels)
    
    
    # Print results and plot confusion matrix
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    class_labels = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
    plt.figure(figsize=(10,7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # One-hot encode the true labels
    true_labels = label_binarize(y_test, classes=class_labels)

    # One-hot encode the predicted labels
    predicted_labels = label_binarize(y_pred, classes=class_labels)

    # Compute ROC AUC score for each class
    auc_scores = []
    for i in range(len(class_labels)):
        auc_scores.append(roc_auc_score(true_labels[:, i], predicted_labels[:, i]))

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'orange']
    for i, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(true_labels[:, i], predicted_labels[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'ROC curve ({label}) - AUC = {auc_scores[i]:.2f}')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()



    