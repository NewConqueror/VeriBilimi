from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt

"""
Bu kod, Agglomerative Clustering (Hiyerarşik Kümeleme) algoritmasını kullanarak örnek bir 
veri kümesi üzerinde dört farklı kümeleme yöntemi (linkage method) uygular ve sonuçları 
görselleştirir.

make_blobs: Örnek veri kümeleri oluşturmak için kullanılır. Bu fonksiyon, belirli sayıda 
merkez etrafında dağılan noktaları üretir.
AgglomerativeClustering: Hiyerarşik kümeleme yapmak için kullanılan bir modeldir.
dendrogram ve linkage: Dendrogram (ağaç diyagramı) oluşturmak ve veriler arasındaki 
mesafeleri hesaplamak için kullanılır. Dendrogram, hiyerarşik kümeleme yapısının görsel 
bir temsilidir.

make_blobs: 300 örnek noktadan oluşan, dört merkez etrafında toplanmış bir veri kümesi 
oluşturur. cluster_std=0.6 ile her bir kümenin standart sapması belirlenir. 
random_state=42 kullanılarak sonuçların tekrarlanabilir olması sağlanır.
plt.scatter: Üretilen veriyi 2 boyutlu bir grafikte noktalar halinde gösterir.
"""

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Ornek Veri")

linkage_methods = ["ward", "single", "average", "complete"]

plt.figure()
for i, linkage_method in enumerate(linkage_methods, 1):
    
    model = AgglomerativeClustering(n_clusters=4, linkage=linkage_method)
    cluster_labels = model.fit_predict(X)
    
    plt.subplot(2, 4, i)
    plt.title(f"{linkage_method.capitalize()} Linkage Dendogram")
    dendrogram(linkage(X, method = linkage_method), no_labels = True)
    plt.xlabel("Veri noktalari")
    plt.ylabel("uzaklik")
    
    plt.subplot(2, 4, i + 4)
    plt.scatter(X[:, 0], X[:, 1], c = cluster_labels, cmap = "viridis")
    plt.title(f"{linkage_method.capitalize()} Linkage Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")

"""
linkage_methods: Hiyerarşik kümeleme algoritmasında kullanılan dört farklı linkage 
(bağlantı) yöntemini içeren bir liste:

ward: Gruplar arasındaki toplam varyansı en aza indirmeye çalışır. 
Bu yöntem genellikle daha kompakt ve aynı büyüklükteki kümeler oluşturur.
single: İki küme arasındaki en kısa mesafeyi dikkate alır. 
Bu yöntem bazen zincir benzeri uzun ve ince kümeler oluşturabilir.
average: İki küme arasındaki ortalama mesafeyi kullanır.
complete: İki küme arasındaki en uzun mesafeyi dikkate alır. 
Bu yöntem genellikle daha kompakt kümeler oluşturur.
for i, linkage_method in enumerate(linkage_methods, 1): Bu döngü, listedeki dört farklı 
linkage yöntemini sırayla dener ve her bir yöntemi için kümeleme işlemi gerçekleştirir.

AgglomerativeClustering(n_clusters=4, linkage=linkage_method): Her bir linkage yöntemi için 
hiyerarşik kümeleme modelini tanımlar ve dört küme oluşturulmasını ister.

plt.subplot(2, 4, i): 2x4'lük bir ızgarada, her bir linkage yöntemi için dendrogramı 
görselleştireceğimiz bir alt grafik oluşturur.

dendrogram(linkage(X, method=linkage_method), no_labels=True): Verilerin hiyerarşik 
yapısını gösteren bir dendrogram oluşturur. no_labels=True parametresi, 
veri noktalarının etiketlerini gizler.

plt.subplot(2, 4, i + 4): Aynı ızgarada, her bir linkage yöntemi için kümeleme sonuçlarını 
görselleştireceğimiz bir alt grafik oluşturur.

plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap="viridis"): Kümeleme sonuçlarını, 
veri noktalarını küme etiketlerine göre renklendirerek görselleştirir.


Bu kod, dört farklı hiyerarşik kümeleme yöntemini (linkage method) kullanarak aynı 
veri kümesi üzerinde kümeleme işlemi gerçekleştirir ve bu yöntemlerin sonuçlarını 
hem dendrogram (ağaç diyagramı) hem de iki boyutlu scatter plot (dağılım grafiği) olarak 
görselleştirir. Bu sayede, farklı linkage yöntemlerinin aynı veri kümesi üzerinde nasıl 
farklı sonuçlar verdiği kolayca karşılaştırılabilir.
"""