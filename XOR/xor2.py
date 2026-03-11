"""
XOR Problemi - TensorFlow/Keras ile Çok Katmanlı Sinir Ağı
==========================================================
Gizli katmanlı bir ağ ile XOR'u Keras API kullanarak eğitiyorum.
"""

# Uyarıları azalt (TensorFlow import'tan önce)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# XOR veri seti: 4 girdi ve beklenen çıktılar
# (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Sequential model: katmanları sırayla ekliyorum
model = Sequential([
    Input(shape=(2,)),                  # Girdi: 2 özellik (x1, x2)
    Dense(2, activation="sigmoid"),     # Gizli katman: 2 nöron
    Dense(1, activation="sigmoid"),     # Çıkış: 1 nöron (0-1)
])

# Kayıp fonksiyonu: ikili sınıflandırma için binary_crossentropy
# Optimizer: Adam. Metrik: accuracy
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Eğitim: 2000 epoch (XOR için yeterli), ilerleme çubuğu açık (verbose=1)
print("Eğitim başlıyor...")
history = model.fit(X, y, epochs=2000, verbose=1)
print("Eğitim bitti.\n")

# Tahmin: NumPy dizisine çeviriyorum (yazdırma hatası olmasın diye)
predictions = np.asarray(model.predict(X, verbose=0)).reshape(-1)

# Sonuçları yazdırıyorum
print("=" * 45)
print("XOR - TensorFlow/Keras Sonuçları")
print("=" * 45)
print("Girdi    -> Gerçek | Tahmin (ham) | Tahmin (0/1)")
print("-" * 45)
for i in range(len(X)):
    p_val = float(predictions[i])
    tahmin_01 = 1 if p_val >= 0.5 else 0
    print(f"{list(X[i])} ->   {int(y[i,0])}    | {p_val:.4f}      | {tahmin_01}")

dogruluk = np.mean((predictions >= 0.5) == y.ravel())
print("-" * 45)
print(f"Doğruluk: {dogruluk:.2%}")
print("=" * 45)

# Eğitim sürecinde loss ve accuracy grafiği
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(history.history["loss"], color="C0")
axes[0].set_title("Eğitim Kaybı (Loss)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

axes[1].plot(history.history["accuracy"], color="C1")
axes[1].set_title("Eğitim Doğruluğu (Accuracy)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")

plt.tight_layout()
plt.savefig("xor2_egitim.png", dpi=120)
plt.show()

print("\nEğitim grafiği 'xor2_egitim.png' olarak kaydedildi.")

# Karar sınırı görselleştirmesi (xor.py'deki gibi)
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
Z = np.asarray(model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0))
Z = (Z >= 0.5).astype(int).reshape(xx.shape)

plt.figure(figsize=(5, 4))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=100, edgecolors="black", cmap="RdYlBu")
plt.title("Keras MLP - XOR Karar Sınırı")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.tight_layout()
plt.savefig("xor2_karar_siniri.png", dpi=120)
plt.show()

print("Karar sınırı görseli 'xor2_karar_siniri.png' olarak kaydedildi.")
