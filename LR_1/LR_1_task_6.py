import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utilities import visualize_classifier

data = np.loadtxt('data_multivar_nb.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1].astype(int)


nb = GaussianNB().fit(X, y)

svm = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X, y)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)
nb_h = GaussianNB().fit(X_tr, y_tr)
svm_h = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_tr, y_tr)

def report(model_name, y_true, y_pred):
    print(f"{model_name:>6} | acc={accuracy_score(y_true, y_pred):.3f} "
          f"prec={precision_score(y_true, y_pred, average='weighted'):.3f} "
          f"rec={recall_score(y_true, y_pred, average='weighted'):.3f} "
          f"f1={f1_score(y_true, y_pred, average='weighted'):.3f}")

print("Hold-out metrics (20% test):")
report("NB",  y_te, nb_h.predict(X_te))
report("SVM", y_te, svm_h.predict(X_te))

print("\n3-fold CV (weighted F1):")
for name, model in [("NB", GaussianNB()), ("SVM", SVC(kernel='rbf', C=1.0, gamma='scale'))]:
    scores = cross_val_score(model, X, y, scoring='f1_weighted', cv=3)
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")


png_nb  = visualize_classifier(nb,  X, y, title="GaussianNB — decision regions (all data)")
png_svm = visualize_classifier(svm, X, y, title="SVM (RBF) — decision regions (all data)")
print("Saved:", png_nb, "and", png_svm)