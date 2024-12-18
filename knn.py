import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,ConfusionMatrixDisplay
df = pd.read_csv("C:\\Users\\AKSHAYA\\Downloads\\spam (1).csv",encoding='ISO-8859-1')
X = df['v2'].values  
y = df['v1'].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)  # Fit and transform the training data
X_test = cv.transform(X_test)        # Only transform
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_predict_knn = knn_classifier.predict(X_test)
print("K-Nearest Neighbors (KNN) Classification Report:")
print(classification_report(y_test, y_predict_knn))
knn_accuracy = accuracy_score(y_test, y_predict_knn)
print(f"KNN Accuracy: {knn_accuracy:.2f}")
cm_knn = confusion_matrix(y_test, y_predict_knn, labels=['ham', 'spam'])
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=['ham', 'spam'])
disp_knn.plot(cmap=plt.cm.Blues)
plt.title("K-Nearest Neighbors (KNN) - Confusion Matrix")
plt.show()
plt.figure(figsize=(6, 6))
pd.Series(y_predict_knn).value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["#66c2a5", "#fc8d62"])
plt.title("K-Nearest Neighbors (KNN) - Spam vs Ham Prediction Distribution")
plt.ylabel('')
plt.show()
