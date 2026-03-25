import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("heart_stacking.csv")
X = df[['Age', 'Cholesterol', 'MaxHeartRate', 'RestingBP']]
X = df[['Age', 'Cholesterol', 'MaxHeartRate', 'RestingBP']].copy()
y = df['HeartDisease']

X['Age'] = X['Age'] + np.random.randint(-3, 3, size=X.shape[0])
X['Cholesterol'] = X['Cholesterol'] + np.random.randint(-10, 10, size=X.shape[0])
X['MaxHeartRate'] = X['MaxHeartRate'] + np.random.randint(-5, 5, size=X.shape[0])
X['RestingBP'] = X['RestingBP'] + np.random.randint(-5, 5, size=X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

lr = LogisticRegression(max_iter=200, C=0.5)
svm = SVC(probability=True, C=0.5, kernel='linear')
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)

lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
dt.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_dt = dt.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
acc_svm = accuracy_score(y_test, y_pred_svm)
acc_dt = accuracy_score(y_test, y_pred_dt)

estimators = [
    ('lr', LogisticRegression(max_iter=200, C=0.5)),
    ('svm', SVC(probability=True, C=0.5, kernel='linear')),
    ('dt', DecisionTreeClassifier(max_depth=3, min_samples_leaf=5))
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(C=0.5)
)

stack_model.fit(X_train, y_train)

y_pred_stack = stack_model.predict(X_test)
acc_stack = accuracy_score(y_test, y_pred_stack)

models = ['Logistic Regression', 'SVM', 'Decision Tree', 'Stacking']
accuracies = [acc_lr, acc_svm, acc_dt, acc_stack]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()

print(acc_lr, acc_svm, acc_dt, acc_stack)
