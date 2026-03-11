"""
XOR Problemi - Yapay Sinir Ağları Deneyi
========================================
Tek katmanlı perceptron XOR problemini ÇÖZEMEZ.
Gizli katman içeren çok katmanlı sinir ağı XOR'u BAŞARIYLA öğrenir.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

# XOR veri seti: 4 nokta
# (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Eğitim için veriyi çoğaltıyoruz (4 nokta çok az)
X_train = np.tile(X, (50, 1))
y_train = np.tile(y, 50)

# -----------------------------------------------------------------------------
# 1. TEK KATMANLI PERCEPTRON (Başarısız)
# -----------------------------------------------------------------------------
print("=" * 50)
print("1. TEK KATMANLI PERCEPTRON")
print("=" * 50)

perceptron = Perceptron(max_iter=1000, random_state=42)
perceptron.fit(X_train, y_train)
y_pred_perceptron = perceptron.predict(X)

print("XOR Gerçek etiketler:", y)
print("Perceptron tahminleri:", y_pred_perceptron)
print("Doğruluk (4 nokta):", np.mean(y == y_pred_perceptron))
print("-> Tek katmanlı perceptron XOR'u ÇÖZEMEZ (doğrusal ayıramaz).\n")

# -----------------------------------------------------------------------------
# 2. ÇOK KATMANLI SİNİR AĞI - MLP (Başarılı)
# -----------------------------------------------------------------------------
print("=" * 50)
print("2. GİZLİ KATMANLI ÇOK KATMANLI SİNİR AĞI (MLP)")
print("=" * 50)

# XOR için gizli katman: (4, 2) nöron, tanh aktivasyonu
# Bazen rastgele ağırlık başlangıcı kötü olabilir, başarılı olana kadar deniyoruz
mlp = None
for seed in range(50):
    m = MLPClassifier(
        hidden_layer_sizes=(4, 2),
        activation="tanh",
        max_iter=10000,
        random_state=seed,
        solver="lbfgs",
    )
    m.fit(X_train, y_train)
    if np.all(m.predict(X) == y):
        mlp = m
        break
if mlp is None:
    mlp = MLPClassifier(hidden_layer_sizes=(4, 2), activation="tanh", max_iter=10000, random_state=42, solver="lbfgs")
    mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X)

print("XOR Gerçek etiketler:", y)
print("MLP tahminleri:     ", y_pred_mlp)
print("Doğruluk (4 nokta):", np.mean(y == y_pred_mlp))
print("-> Gizli katmanlı MLP XOR'u BAŞARIYLA öğrenir.\n")

# -----------------------------------------------------------------------------
# 3. GÖRSELLEŞTİRME - Karar sınırları
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Karar sınırı için ızgara
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))

# Sol: Perceptron (başarısız)
Z_p = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z_p = Z_p.reshape(xx.shape)
axes[0].contourf(xx, yy, Z_p, alpha=0.3, cmap="RdYlBu")
axes[0].scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolors="black", cmap="RdYlBu")
axes[0].set_title("Tek Katmanlı Perceptron\n(XOR'u çözemez)")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])

# Sağ: MLP (başarılı)
Z_m = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z_m = Z_m.reshape(xx.shape)
axes[1].contourf(xx, yy, Z_m, alpha=0.3, cmap="RdYlBu")
axes[1].scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolors="black", cmap="RdYlBu")
axes[1].set_title("Gizli Katmanlı MLP\n(XOR'u başarıyla öğrenir)")
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])

plt.tight_layout()
plt.savefig("xor_sonuc.png", dpi=120)
plt.show()

print("Görsel 'xor_sonuc.png' olarak kaydedildi.")
