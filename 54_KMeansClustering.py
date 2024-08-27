from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

"""
Bu kod, K-Means algoritmasını kullanarak örnek bir veri kümesindeki kümeleri tespit eder 
ve bu kümeleri görselleştirir.

make_blobs: Örnek veri kümeleri oluşturmak için kullanılan bir fonksiyon.
Bu fonksiyon, belirli sayıda merkez etrafında dağılan noktaları üretir.
KMeans: Veri kümelerini belirli sayıda küme olarak ayırmak için kullanılan bir algoritma.
"""

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

"""
make_blobs: n_samples=300 parametresi ile 300 örnek nokta oluşturur. 
centers=4 ile dört merkez etrafında toplanan noktalar üretilir. cluster_std=0.6 her bir 
kümenin standart sapmasını belirler, yani noktaların merkezlerinden ne kadar uzaklaştığını 
gösterir. random_state=42 ise sonuçların tekrarlanabilir olmasını sağlar.
"""

plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Ornek Veri")

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

labels = kmeans.labels_

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = "viridis")

"""
kmeans.labels_: Her veri noktası için hangi kümeye ait olduğunu belirten etiketleri döndürür. 
Bu etiketler, 0 ile 3 arasında değerler alır (çünkü 4 küme var).

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis"): Veri noktalarını kümelere göre 
farklı renklere boyayarak grafiğe çizer. c=labels parametresi, noktaların renklerinin hangi 
kümeye ait olduklarına göre belirlenmesini sağlar. cmap="viridis" ise kullanılan 
renk haritasını belirler.
"""

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = "red", marker = "X")
plt.title("K-Means")

"""
kmeans.cluster_centers_: Her kümenin merkezlerini döndürür. 
Bu merkezler, K-Means algoritmasının bulduğu en optimal merkez noktalarıdır.

plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X"): Küme merkezlerini kırmızı 
"X" işaretleriyle grafikte gösterir.
"""

"""
Bu kod, rastgele oluşturulmuş bir veri kümesindeki noktaları K-Means algoritması kullanarak 
dört kümeye ayırır ve bu kümeleri farklı renklerde görselleştirir. 
Ayrıca, her kümenin merkezini de kırmızı "X" işaretleriyle gösterir. 
Bu işlem, verilerin belirli gruplar halinde nasıl toplandığını ve K-Means algoritmasının 
bu grupları nasıl tanımladığını anlamak için kullanılır.
"""