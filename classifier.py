from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_svm_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    params = {'C': [0.1, 1, 10, 100], 'gamma': ['auto', 'scale', 0.001, 0.01, 0.1], 'kernel': ['rbf', 'linear', 'poly'], 'class_weight': [None, 'balanced']}
    svm = SVC()
    clf = GridSearchCV(svm, params, cv=3)
    clf.fit(X_train, y_train)
    # Display the best estimator
    print("Best model:", clf.best_estimator_)
    # Display the best score
    print("Best cross-validation score:", clf.best_score_)
    # Display the best parameters
    print("Best parameters:", clf.best_params_)
    y_pred = clf.predict(X_test)
    return y_test, y_pred, clf
