import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    Y_train = np.load('Y_train.npy')
    Y_test = np.load('Y_test.npy')

    logistic_regression = LogisticRegression(solver='liblinear', C=0.001, class_weight='balanced')
    svm = SVC(kernel='linear') # Support Vector Machine

    logistic_regression.fit(X_train, Y_train)
    svm.fit(X_train, Y_train)

    predictions_lr = logistic_regression.predict(X_test)
    predictions_svm = svm.predict(X_test)

    accuracy_lr = accuracy_score(Y_test, predictions_lr)
    accuracy_svm = accuracy_score(Y_test, predictions_svm)

    print('Logistic Regression Accuracy:', accuracy_lr)
    print('SVM Accuracy:', accuracy_svm)
