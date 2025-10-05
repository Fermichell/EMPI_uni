import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1].astype(int)

classifier = GaussianNB()
classifier.fit(X, y)
y_pred = classifier.predict(X)

accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")

png_all = visualize_classifier(classifier, X, y, title="GaussianNB — decision regions (all data)")
print("Saved figure:", png_all)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)
classifier_new = GaussianNB().fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)
acc_test = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the new classifier =", round(acc_test, 2), "%")

png_test = visualize_classifier(classifier_new, X_test, y_test, title="GaussianNB — decision regions (test split)")
print("Saved figure:", png_test)

# Cross-validation (3-fold)
num_folds = 3
for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
    scores = cross_val_score(GaussianNB(), X, y, scoring=metric, cv=num_folds)
    print(f"{metric}: {scores.mean():.3f} (+/- {scores.std():.3f})")
