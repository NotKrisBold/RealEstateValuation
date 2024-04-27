from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# fetch dataset
real_estate_valuation = fetch_ucirepo(id=477)

# Inizializzazione del database
X = pd.DataFrame(real_estate_valuation.data.features, columns=real_estate_valuation.feature_names)
y = pd.Series(real_estate_valuation.data.targets.values.flatten(), name='house price of unit area')

# Calcolo della dimensione approssimativa di ciascun intervallo
classes = ['basso', 'medio', 'alto']

# Trasformazione della variabile target in tre categorie di valore
y_multiclass = pd.qcut(y, q=len(classes), labels=classes)
print(y_multiclass.value_counts())

# Training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y_multiclass, test_size=0.2, random_state=42)

# Modello di classificazione multiclasse
classifier = LogisticRegression(max_iter=350*len(classes))

# Addestramento del modello sul training set
classifier.fit(X_train, y_train)

# Valutazione del modello sul test set
y_pred = classifier.predict(X_test)

# Metriche di valutazione
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Stampare il classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, labels=classes, zero_division=0))

# Calcolare la confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix with ordered classes
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
