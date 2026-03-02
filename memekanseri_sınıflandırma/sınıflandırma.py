import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Bu çalışmada meme kanserini iki sınıfa ayırmak için sınıflandırma yapıyorum.
# Veri seti olarak sklearn içindeki Breast Cancer Wisconsin (Diagnostic) veri setini kullanıyorum.

# Veri setini sklearn içindeki hazır Wisconsin veri setinden yüklüyorum.
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
# Bu veri setinde 0=malignant (kötü huylu), 1=benign (iyi huylu) etiketlerini kullanıyorum.
df["diagnosis"] = data.target

print("Veri seti: Breast Cancer Wisconsin Dataset (sklearn)")
print("Veri boyutu:", df.shape, "(30 özellik + 1 sınıf)")
print("Sınıflar:", data.target_names)  
print("Sınıf dağılımı (0=Malignant, 1=Benign):")
print(df["diagnosis"].value_counts())
print(df.head())

# Özellik matrisi X ve hedef vektörü y
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

print("X shape:", X.shape)
print("Y unique:", y.unique())

# Veriyi eğitim ve test olarak %80 eğitim, %20 test olacak şekilde ayırılır.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train-test bölme başarılı")

#StandardScaler ile ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Sınıflandırma için Logistic Regression modeli kullanıldı.
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Sonuçlar ---")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("\nClassification Report:")

print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Confusion matrix'i görselleştirme 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"],
)
plt.ylabel("Gerçek")
plt.xlabel("Tahmin")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
plt.show()

