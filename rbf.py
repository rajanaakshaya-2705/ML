import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_classes=4,  # Reduced to meet constraints
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_clusters_per_class=1,
    random_state=42
)
model = SVR(kernel='rbf', C=1, epsilon=0.1)
model.fit(X, y)
predictions = model.predict(X)
predicted_classes = np.round(predictions)
accuracy = accuracy_score(y, predicted_classes)
print(f"Accuracy: {accuracy:.2f}")
plt.scatter(X[:, 0], X[:, 1], c=predicted_classes, cmap='viridis', edgecolor='k')
plt.colorbar(label='Predicted Class')
plt.title(f"SVR Predictions with make_classification Dataset (Accuracy: {accuracy:.2f})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
