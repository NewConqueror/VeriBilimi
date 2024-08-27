import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

"""
Bu kod, müşteri segmentasyonu yapmak amacıyla K-Means ve hiyerarşik kümeleme (dendrogram) 
yöntemlerini kullanarak bir veri kümesi üzerinde kümeleme analizi gerçekleştirir.

pd.read_pickle("4_5_GozetimsizOgrenmeDurumCalismasi"): Veriyi bir pickle dosyasından yükler. 
Bu dosya, müşteri segmentasyonu için kullanılan gelir ve harcama skorları gibi özellikleri 
içermektedir.

X[:, 0] ve X[:, 1]: Verinin her iki özelliği (income ve spending score) üzerinde mutlak 
değer alma ve bir ölçekleme işlemi gerçekleştirir. Bu, negatif değerlerin pozitif 
yapılmasını ve tüm değerlerin pozitif aralığa getirilmesini sağlar
"""

data = pd.read_pickle("4_5_GozetimsizOgrenmeDurumCalismasi")
X = data.values

X[:, 0] = np.abs(2*min(X[:, 0])) + X[:, 0]
X[:, 1] = np.abs(2*min(X[:, 1])) + X[:, 1]

plt.figure()
plt.scatter(X[:, 0], X[:, 1], s = 50, alpha = 0.7, edgecolors="k")
plt.title("Musteri Segmentasyonu")
plt.xlabel("income")
plt.ylabel("spending score")

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

plt.figure(figsize = (15,6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c = cluster_labels, s = 50, alpha = 0.7, edgecolors="k")
plt.title("Kmean - Musteri Segmentasyonu")
plt.xlabel("income")
plt.ylabel("spending score")

linkage_matrix = linkage(X, method = "ward")
plt.subplot(1, 2, 2)
dendrogram(linkage_matrix)
plt.title("Dendrogram - Musteri Segmentasyonu")
plt.xlabel("Veri Noktari")
plt.ylabel("Uzaklik")

"""
kmeans.cluster_centers_: Her kümenin merkezlerini döndürür.

linkage_matrix = linkage(X, method="ward"): Hiyerarşik kümeleme algoritmasını kullanarak 
bir bağlantı matrisi oluşturur. Burada ward yöntemi kullanılır, bu yöntem iki kümeyi 
birleştirirken toplam varyansı minimumda tutmayı amaçlar.
plt.subplot(1, 2, 2): İkinci bir alt grafik oluşturur. Bu grafik, dendrogramı gösterecektir.
dendrogram(linkage_matrix): Veri noktalarının birbirine olan uzaklıklarına göre oluşturulan 
dendrogramı çizer. Dendrogram, kümeleme işleminin hiyerarşik yapısını görselleştirir ve 
küme sayısını belirlemek için kullanılabilir.


Bu kod, K-Means ve hiyerarşik kümeleme yöntemlerini kullanarak müşteri segmentasyonu yapar 
ve bu segmentlerin görselleştirilmesini sağlar. İlk grafik, verilerin orijinal halini 
gösterirken; ikinci grafikler, K-Means ile belirlenen kümeleri ve hiyerarşik kümeleme ile 
oluşturulan dendrogramı gösterir. Bu yöntemler, müşterileri gelir ve harcama skorlarına göre 
benzer davranışları olan gruplara ayırmaya yarar.
"""
