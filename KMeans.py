import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from scipy.stats import mode
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


plt.ion()

print("=" * 50)
print("1. IRIS VERİ SETİNİ YÜKLEME")
print("=" * 50)


iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names


df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(f"Toplam örnek sayısı: {X.shape[0]}")
print(f"Özellik sayısı: {X.shape[1]}")
print(f"Özellikler: {feature_names}")
print(f"Sınıflar: {target_names}")
print("\nVeri setinin ilk 5 örneği:")
print(df.head())

print("\nHedef sınıfların dağılımı:")
print(df['species'].value_counts())

print("\nTemel istatistikler:")
print(df.iloc[:, :4].describe())


plt.figure(figsize=(12, 10))
plt.suptitle('Özelliklerin Dağılımları', fontsize=16)

for i in range(4):
    plt.subplot(2, 2, i+1)
    for j in range(3):
        plt.hist(X[y==j, i], bins=10, alpha=0.5, label=target_names[j])
    plt.title(feature_names[i])
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
print("Her özelliğin her sınıf için dağılımını gösteren histogramlar oluşturuldu.")
input("Devam etmek için Enter tuşuna basın...")

print("\n" + "=" * 50)
print("2. VERİ ÖN İŞLEME: STANDARTLAŞTIRMA")
print("=" * 50)

print("Standartlaştırma öncesi veri örneği:")
print(df.iloc[:5, :4])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=feature_names)

print("\nStandartlaştırma sonrası veri örneği:")
print(df_scaled.head())

print("\nStandartlaştırma sonrası temel istatistikler:")
print(df_scaled.describe())
print("Not: Ortalama ~0, standart sapma ~1 olduğunu görebilirsiniz.")

plt.figure(figsize=(14, 6))
plt.suptitle('Standartlaştırma Öncesi vs Sonrası', fontsize=16)

plt.subplot(1, 2, 1)
plt.hist(df['sepal length (cm)'], bins=20, alpha=0.5, label='Orijinal')
plt.hist(df_scaled['sepal length (cm)'], bins=20, alpha=0.5, label='Standartlaştırılmış')
plt.title('Sepal Length')
plt.grid(linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(df['petal length (cm)'], bins=20, alpha=0.5, label='Orijinal')
plt.hist(df_scaled['petal length (cm)'], bins=20, alpha=0.5, label='Standartlaştırılmış')
plt.title('Petal Length')
plt.grid(linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
print("Standartlaştırma öncesi ve sonrası karşılaştırma grafikleri oluşturuldu.")
input("Devam etmek için Enter tuşuna basın...")

print("\n" + "=" * 50)
print("3. OPTIMAL KÜME SAYISINI BELİRLEME (ELBOW METHOD)")
print("=" * 50)

inertia = []
silhouette_scores = []

from sklearn.metrics import silhouette_score

print("K değeri\tInertia\t\tDeğişim Yüzdesi")
print("-" * 40)

prev_inertia = None
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    curr_inertia = kmeans.inertia_
    inertia.append(curr_inertia)

    if prev_inertia is not None:
        change = (prev_inertia - curr_inertia) / prev_inertia * 100
        print(f"{k}\t\t{curr_inertia:.2f}\t\t{change:.2f}%")
    else:
        print(f"{k}\t\t{curr_inertia:.2f}\t\t-")

    prev_inertia = curr_inertia

    if k > 1:
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(0)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.xlabel('Küme Sayısı (K)')
plt.ylabel('Inertia (Toplam Kare Hatası)')
plt.title('Elbow Metodu ile Optimal K Değeri')
plt.grid(True, linestyle='--', alpha=0.7)
plt.annotate('Dirsek Noktası', xy=(3, inertia[2]),
             xytext=(4, inertia[2] + 50),
             arrowprops=dict(facecolor='red', shrink=0.05))

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores[1:], marker='o', linestyle='--')
plt.xlabel('Küme Sayısı (K)')
plt.ylabel('Silhouette Skoru')
plt.title('Silhouette Skoru ile Optimal K Değeri')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
print("Elbow Method ve Silhouette Skoru grafikleri oluşturuldu.")
print("Grafiklerden görüleceği üzere optimal küme sayısı k=3 olarak belirlenmiştir.")
input("Devam etmek için Enter tuşuna basın...")

print("\n" + "=" * 50)
print("4. K-MEANS KÜMELEMESİ UYGULAMA (k=3)")
print("=" * 50)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
labels = kmeans.labels_

unique, counts = np.unique(labels, return_counts=True)
print("Kümelerin eleman sayıları:")
for i, (u, c) in enumerate(zip(unique, counts)):
    print(f"Küme {u}: {c} eleman")

centers = kmeans.cluster_centers_
print("\nKüme merkezleri (standartlaştırılmış):")
centers_df = pd.DataFrame(centers, columns=feature_names)
print(centers_df)

centers_orig = scaler.inverse_transform(centers)
centers_orig_df = pd.DataFrame(centers_orig, columns=feature_names)
print("\nKüme merkezleri (orijinal ölçek):")
print(centers_orig_df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(centers)

explained_variance = pca.explained_variance_ratio_
print(f"\nPCA'nın açıkladığı varyans: {explained_variance[0]:.2f}, {explained_variance[1]:.2f}")
print(f"Toplam açıklanan varyans: {sum(explained_variance):.2f}")

plt.figure(figsize=(7, 6))
colors = ['#440154', '#21918c', '#fde725']

for i in range(3):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1],
                c=colors[i], label=f'Küme {i}',
                edgecolor='k', s=50, alpha=0.7)

plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
            c='red', marker='X', s=200, label='Küme Merkezleri')

plt.xlabel('PCA Bileşen 1')
plt.ylabel('PCA Bileşen 2')
plt.title('K-means Kümeleri (k=3)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'),
             label='Küme Rengi')
plt.show()
print("K-means kümeleri PCA kullanılarak görselleştirildi.")
input("Devam etmek için Enter tuşuna basın...")

print("\n" + "=" * 50)
print("5. GERÇEK ETİKETLERLE KARŞILAŞTIRMA")
print("=" * 50)

crosstab = pd.crosstab(y, labels, rownames=['Gerçek'], colnames=['K-means'])
print("Gerçek etiketler vs K-means kümeleri:")
print(crosstab)

dominant_class = {}
for i in range(3):
    mask = (labels == i)
    if np.any(mask):
        dominant = mode(y[mask])[0]
        dominant_class[i] = dominant
        print(f"Küme {i}'in baskın sınıfı: {target_names[dominant]}")

matched_labels = np.zeros_like(labels)
for i in range(3):
    mask = (labels == i)
    if np.any(mask):
        matched_labels[mask] = dominant_class[i]

accuracy = accuracy_score(y, matched_labels)
print(f"\nEşleştirilmiş etiketlerle doğruluk: {accuracy:.4f}")

conf_matrix = confusion_matrix(y, matched_labels)
print("\nConfusion Matrix:")
print(conf_matrix)

print("\nSınıflandırma Raporu:")
print(classification_report(y, matched_labels, target_names=list(target_names)))

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for i in range(3):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                c=colors[i], label=target_names[i],
                edgecolor='k', s=50, alpha=0.7)
plt.title('Gerçek Türler')
plt.xlabel('PCA Bileşen 1')
plt.ylabel('PCA Bileşen 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 3, 2)
for i in range(3):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1],
                c=colors[i], label=f'Küme {i}',
                edgecolor='k', s=50, alpha=0.7)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
            c='red', marker='X', s=100, label='Merkezler')
plt.title('Ham K-means Kümeleri')
plt.xlabel('PCA Bileşen 1')
plt.ylabel('PCA Bileşen 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 3, 3)
for i in range(3):
    plt.scatter(X_pca[matched_labels == i, 0], X_pca[matched_labels == i, 1],
                c=colors[i], label=target_names[i],
                edgecolor='k', s=50, alpha=0.7)
plt.title('Eşleştirilmiş K-means Kümeleri')
plt.xlabel('PCA Bileşen 1')
plt.ylabel('PCA Bileşen 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
print("Karşılaştırma grafikleri oluşturuldu.")
input("Devam etmek için Enter tuşuna basın...")

print("\n" + "=" * 50)
print("6. FARKLI KÜME SAYILARI İÇİN KARŞILAŞTIRMA")
print("=" * 50)

k_values = range(2, 6)
accuracies = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    k_labels = kmeans.labels_

    k_matched = np.zeros_like(k_labels)
    for i in range(k):
        mask = (k_labels == i)
        if np.any(mask):
            k_matched[mask] = mode(y[mask])[0]

    acc = accuracy_score(y, k_matched)
    accuracies.append(acc)
    print(f"K={k} için doğruluk: {acc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('Küme Sayısı (K)')
plt.ylabel('Doğruluk')
plt.title('Farklı K Değerleri için Doğruluk')
plt.xticks(k_values)
plt.ylim(0.8, 1.0)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

print("\n" + "=" * 50)
print("7. SONUÇ VE DEĞERLENDİRME")
print("=" * 50)

print("K-means Kümeleme Analizi Sonuçları:")
print(f"1. Optimal küme sayısı: 3 (Elbow metodu ve Silhouette skoruyla doğrulandı)")
print(f"2. K=3 için doğruluk: {accuracy:.4f}")
print(f"3. En iyi doğruluk veren K değeri: {k_values[np.argmax(accuracies)]}, doğruluk: {max(accuracies):.4f}")

print("\nKüme Merkezleri (Orijinal Ölçek):")
for i, row in centers_orig_df.iterrows():
    print(f"Küme {i}: {row.tolist()}")

print("\nK-means kümeleme algoritması, denetimsiz öğrenme yöntemi olmasına rağmen,")
print("Iris veri setindeki gerçek sınıfları %{:.1f} doğrulukla tahmin etmeyi başardı.".format(accuracy*100))

plt.ioff()