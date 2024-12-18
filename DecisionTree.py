 import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
cu = DecisionTreeClassifier(random_state=42,criterion='entropy')
cu.fit(X_train, y_train)
print(f"Accuracy: {cu.score(X_test, y_test):.2f}")
plt.figure(figsize=(40,50))
plot_tree(cu, filled=True, feature_names=iris.feature_names, class_names=list(iris.target_names))
plt.title("Unpruned Decision Tree")
plt.show()
