import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
cl=DecisionTreeClassifier(random_state=42,max_depth=3,min_samples_split=5)
cl.fit(x_train,y_train)
print("Accuracy",cl.score(x_test,y_test))
plt.figure(figsize=(12,12))
plot_tree(cl, filled=True, feature_names=iris.feature_names, class_names=list(iris.target_names))
plt.show()
