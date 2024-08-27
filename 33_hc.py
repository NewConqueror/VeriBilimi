import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:].values

#kmeans

from sklearn.cluster import KMeans

kmeans = KMeans ( n_clusters = 3, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11),sonuclar)
plt.show()

kmeans = KMeans (n_clusters = 4, init='k-means++', random_state= 123)
Y_tahmin= kmeans.fit_predict(X)
print(Y_tahmin)  
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')
plt.title('KMeans')
plt.show()

#HC
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)

plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')
plt.title('HC')
plt.show()

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()

"""
Hiyerarşik bölütleme Hierarchical Clustering

küme sayısının önceden belirlenemediği öngörülemediği durumlarda kullanılır

Agglomerative aşağından yukarıya toplayarak gider

Her veri tek bir küme bölüt ile başlar

En yakın ikişer komşuyu alıp ikişerli küme/bölüt oluşturulur

En yakın iki kümeyi alıp yeni bir bölüt oluşturur

Bir önceki adım tek bir küme bölüt olana kadar devam eder

hepsi bir küme en yakın 2 sini birleştir yeni küme diğer 2 si başka küme böyle böyle
tek küme kalana kadar devam

Divisive yukarıdan aşağıya bölerek gider

başlangıçta Tek küme var o Tek kümeden en yakın ları birleştirip 2 alt küme yapar
sonra o kümelerin içinden en yakınları birleştirip alt küme yapar
hepsi tek bir küme olana kadar devam eder


Mesafe ölçümleri olarak Öklit mesafesi kullanılır Öklit ölçümü kullanılır

Referanslar 

En yakın Noktalar
En uzak Noktalar
Ortalama
Merkezler Arası Mesafe

nereden ölçtüğüne göre değişiklik gösterir en yakın 2 küme dolayısıyla
birleştirme şeklin de değişir

Dendogram kullanırız bunun için dendogram bize hangi kümelerin birleştirildiğini gösterir
mantıklı olan en uzun mesafeden kesmektir kaç çizgiye değersen o kadar Cluster olması
gerektiği anlamına gelir

Mesafe matrisi var

Farklı mesafe stratejileri var

MIN, MAX, GROUP AVERAGE, WARDS METHOD 

Wards methodu WCSS değerleri kullanılarak yapılır

Bu da bize kaç Cluster ın en iyi sonucu vereceğini söyler

"""


"""

Hierarchical clustering (hiyerarşik kümeleme), verileri hiyerarşik bir yapıda gruplandıran 
bir kümeleme yöntemidir. Bu yöntem, verileri kümeleme işlemini adım adım yaparak, 
küçük kümeleri birleştirerek veya büyük kümeleri bölerek oluşturur. 
Hiyerarşik kümeleme, 

özellikle küme sayısının önceden belirlenemediği durumlarda kullanışlıdır.

Hierarchical Clustering Türleri
Aglomeratif Hiyerarşik Kümeleme (Agglomerative Hierarchical Clustering):

Tanım: Bu yöntem, her veri noktasını başlangıçta kendi kümesi olarak alır ve en yakın iki 
kümeyi birleştirerek devam eder. Süreç, tüm veri noktaları tek bir kümede birleşene kadar 
devam eder.
Adımlar:
Her veri noktası kendi kümesi olarak başlar.
En yakın iki küme bulunur ve birleştirilir.
Adım 2, tüm veri noktaları tek bir kümede birleşene kadar tekrarlanır.

Bölücü Hiyerarşik Kümeleme (Divisive Hierarchical Clustering):

Tanım: Bu yöntem, tüm veri noktalarını başlangıçta tek bir küme olarak alır ve küme içindeki 
veri noktalarını bölerek alt kümeler oluşturur. Süreç, her veri noktası kendi kümesi olana 
kadar devam eder.
Adımlar:
Tüm veri noktaları tek bir küme olarak başlar.
En heterojen kümeyi belirleyin ve bölün.
Adım 2, her veri noktası kendi kümesi olana kadar tekrarlanır.
Mesafe Metodları
Hiyerarşik kümeleme, kümeler arasındaki mesafeyi belirlemek için 
farklı yöntemler kullanabilir:

Single Linkage (Minimum Linkage): İki küme arasındaki en yakın veri noktalarının mesafesi.
Complete Linkage (Maximum Linkage): İki küme arasındaki en uzak veri noktalarının mesafesi.
Average Linkage: İki küme arasındaki tüm veri noktalarının ortalama mesafesi.
Centroid Linkage: Kümelerin merkezleri (centroid) arasındaki mesafe.

Dendrogram
Hiyerarşik kümeleme sonuçlarını görselleştirmek için dendrogram adı verilen bir 
ağaç diyagramı kullanılır. Dendrogram, veri noktalarının hangi sırayla ve hangi mesafelerde
birleştirildiğini gösterir. Yatay eksen kümeleri, dikey eksen ise mesafeyi temsil eder.


Açıklamalar
Dendrogram: sch.linkage fonksiyonu ile veri setinin bağlantı matrisi hesaplanır ve 
sch.dendrogram ile dendrogram çizilir. Bu grafik, kümeler arasındaki mesafeleri ve birleşme 
sırasını gösterir.

Agglomeratif Kümeleme: AgglomerativeClustering modeli ile veri seti kümeleme yapılır ve 
labels değişkeni her veri noktasının hangi kümede olduğunu belirtir. Sonuçlar, 
scatter plot ile görselleştirilir.

Avantajlar ve Dezavantajlar
Avantajlar:

Küme sayısını önceden belirlemek gerekmez.
Dendrogram sayesinde küme yapısının görsel analizi yapılabilir.

Dezavantajlar:

Büyük veri setlerinde hesaplama maliyeti yüksektir.
Gürültü ve aykırı değerlerden etkilenebilir.
Hiyerarşik kümeleme,verilerin doğal yapısını anlamak ve kümeleme sonuçlarını görselleştirmek 
için güçlü bir yöntemdir. Dendrogramlar, küme yapısının detaylı bir analizini sunarak, 
veri bilimcilere ve analistlere değerli bilgiler sağlar.

"""
















