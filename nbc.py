import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,ConfusionMatrixDisplay
df = pd.read_csv("/content/spam.csv",encoding='ISO-8859-1')
X = df['v2'].values  
y = df['v1'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)  
X_test = cv.transform(X_test)        
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_predict_knn = knn_classifier.predict(X_test)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
y_predict_nb = nb_classifier.predict(X_test)
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, y_predict_nb))
nb_accuracy = accuracy_score(y_test, y_predict_nb)
print(f"Naive Bayes Accuracy: {nb_accuracy}")
cm_nb = confusion_matrix(y_test, y_predict_nb, labels=['ham', 'spam'])
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=['ham', 'spam'])
disp_nb.plot(cmap=plt.cm.Blues)
plt.title("Naive Bayes - Confusion Matrix")
plt.show()
plt.figure(figsize=(6, 6))
pd.Series(y_predict_nb).value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["#66c2a5", "#fc8d62"])
plt.title("Naive Bayes - Spam vs Ham Prediction Distribution")
plt.ylabel('')
plt.show()
