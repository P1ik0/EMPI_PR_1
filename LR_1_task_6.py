import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score

from utilities import visualize_classifier

# Завантаження даних із вхідного файлу
input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбивка даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Створення та тренування SVM класифікатора
svm_classifier = SVC(kernel='rbf')
svm_classifier.fit(X_train, y_train)
y_test_pred = svm_classifier.predict(X_test)

accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of SVM classifier on test data =", round(accuracy, 2), "%")

# Показники якості SVM (перехресна перевірка)
num_folds = 3
print("\nSVM:")
for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
    values = cross_val_score(SVC(kernel='rbf'), X, y, scoring=metric, cv=num_folds)
    print(metric + ": " + str(round(100 * values.mean(), 2)) + "%")

# Показники якості наївного байєсовського класифікатора для порівняння
print("\nNaive Bayes:")
for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
    values = cross_val_score(GaussianNB(), X, y, scoring=metric, cv=num_folds)
    print(metric + ": " + str(round(100 * values.mean(), 2)) + "%")

# Візуалізація роботи SVM класифікатора
visualize_classifier(svm_classifier, X_test, y_test)
